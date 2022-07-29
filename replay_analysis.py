import os
import sys
import numpy as np
from matplotlib import pyplot
import pickle
from ec_hpc_net import EcHpc
from matplotlib import mathtext
from cusnn import analysis as sa

mathtext.FontConstantsBase = mathtext.ComputerModernFontConstants

pyplot.rcParams['font.family'] = 'sans-serif'
pyplot.rcParams['xtick.direction'] = 'in'
pyplot.rcParams['ytick.direction'] = 'in'
pyplot.rcParams['xtick.major.width'] = 1.0
pyplot.rcParams['ytick.major.width'] = 1.0
pyplot.rcParams['font.size'] = 8
pyplot.rcParams['axes.linewidth'] = 1.0


def cm2inch(value):
    return value/2.54


class TrajectoryEvent:
    def __init__(self, is_replay, t_vec, x, y, mov):
        self.is_replay = is_replay
        self.t_vec = t_vec
        self.x = x
        self.y = y
        self.mov = mov


def replay_decoding(output_dir, ratemap_free: sa.RateMap, replay_group: sa.SpikeData
                    , x_y_hz: np.ndarray, time_window: list, params:dict, tau_decode=0.025, step=0.005,
                    jump_lim=0.5, least_path=0.8, least_step=10, pc_lim=0.3):
    """
    :param output_dir:
    :param ratemap_free:
    :param replay_group:
    :param time_window: [sec]
    :param tau_decode: [sec]
    :param step: [sec]
    :param jump_lim: [m]
    :param least_path: [m]
    :param least_step: [step]
    :param pc_lim [Hz]
    :return:
    """

    # nan -> 0 for this analysis
    ratemap_free.rate[np.isnan(ratemap_free.rate)] = 0.0

    # bayesian decoding parameters
    start_frame, end_frame = int(time_window[0] / step), int(np.ceil(time_window[1] / step))
    t_vec = np.array([frame * step for frame in range(start_frame, end_frame)])  # sec
    place_exp_factor = np.exp(-tau_decode * np.sum(ratemap_free.rate, axis=0))

    # post prob list
    post_prob_list = []
    event_list = []
    x, y = np.zeros(t_vec.size), np.zeros(t_vec.size)

    # position estimation by Bayesian decoding
    for i, t in enumerate(range(start_frame, end_frame)):
        cnt = 0
        l, r = np.searchsorted(replay_group.spike_timing, t * step), np.searchsorted(replay_group.spike_timing, t * step + tau_decode)
        window_raster = replay_group.cell_id[l:r] - replay_group.member[0]
        post_p = np.ones(ratemap_free.X.shape, dtype=np.float64)
        for cell_num in window_raster:
            if x_y_hz[cell_num, 2] > pc_lim:  # exclude non place cells
                cnt += 1
                post_p *= ratemap_free.rate[cell_num] + 10**-9
                post_p /= np.sum(post_p)
        if cnt:
            post_p *= place_exp_factor
            post_p /= np.sum(post_p)
            post_prob_list.append(post_p)
            x[i], y[i] = np.sum(ratemap_free.X * post_p), np.sum(ratemap_free.Y * post_p)   # weighted mean
        else:
            post_prob_list.append(post_p * 0.0)
            x[i], y[i] = 10**9, 10**9
        del post_p

    # calc x, y position and movement of each step
    L_seq = len(post_prob_list)
    mov = np.sqrt(np.power(np.diff(x), 2) + np.power(np.diff(y), 2))  # [meter]

    # truncation of sequence
    # find longest sequence without jump(movement more than jump_lim from the previous frame)
    mov_jump = np.zeros(mov.size, dtype=np.bool)
    for i, m in enumerate(mov):
        if m > jump_lim:
            mov_jump[i] = np.True_
            if i != len(mov)-1:
                mov_jump[i+1] = np.True_
    if params["task"] == "linear_track":
        mov_zero_cross_x = np.insert(np.diff(np.diff(x) <= 0), 0, np.False_)
        section_edge = np.where(mov_jump | mov_zero_cross_x)[0]
    else:
        mov_zero_cross_x = np.insert(np.diff(np.diff(x) <= 0), 0, np.False_)
        mov_zero_cross_y = np.insert(np.diff(np.diff(y) <= 0), 0, np.False_)
        section_edge = np.where(mov_jump | mov_zero_cross_x | mov_zero_cross_y)[0]

    # Determine if it is a hippocampal replay.
    section_edge = np.concatenate(([0], section_edge, [L_seq - 1]))
    section_length = np.diff(section_edge)
    indexes = np.where(np.max(section_length) == section_length)[0]
    for idx in indexes:
        l_edge, t_edge = section_edge[idx], section_edge[idx + 1] + 1
        st_end = x[t_edge-1] - x[l_edge]
        if (t_edge - l_edge) >= least_step and np.sum(mov[l_edge:t_edge - 1]) >= least_path and st_end > 0:
            print("left = %.3f, right = %.3f" % (time_window[0], time_window[1]))
            print("L_seq = %d, sum(mov) = %.2f" % (t_edge - l_edge, float(np.sum(mov[l_edge:t_edge - 1]))))
            event_list.append(TrajectoryEvent(True, t_vec[l_edge:t_edge], x[l_edge:t_edge], y[l_edge:t_edge], mov[l_edge:t_edge - 1]))
        else:
            event_list.append(TrajectoryEvent(False, t_vec[l_edge:t_edge], x[l_edge:t_edge], y[l_edge:t_edge], mov[l_edge:t_edge - 1]))

        # If it is determined to be a replay, a figure is output.
        if event_list[-1].is_replay is True:

            # save replay event data
            os.makedirs(output_dir, exist_ok=True)
            with open(output_dir + "/replay%.3fto%.3f.pickle" % (event_list[-1].t_vec[0], event_list[-1].t_vec[-1]), "wb") as f:
                pickle.dump(event_list[-1], f)

            # make directory
            event = event_list[-1]

            # draw posterior probability
            base = ratemap_free.X * 0.0
            for i, post_prob in enumerate(post_prob_list[l_edge:t_edge]):
                if i % 10 == 0:
                    pyplot.figure(figsize=(7, 10))
                base = base + post_prob
                pyplot.subplot(10, 1, (i % 10) + 1)
                pyplot.pcolor(ratemap_free.X, ratemap_free.Y, post_prob, cmap=pyplot.cm.afmhot)
                pyplot.ylabel("%.2f" % event.t_vec[i])
                pyplot.tick_params(labelbottom=False, bottom=False, left=False, labelleft=False)
                pyplot.box(False)
                pyplot.axis("equal")
                if i % 10 == 9 or i == (t_edge - l_edge) - 1:
                    begin, end = (i - 9) * step + time_window[0] + l_edge * step, i * step + time_window[0] + l_edge * step
                    pyplot.savefig(output_dir + "/at%.3fto%.3fsec.png" % (begin, end))
                    pyplot.close()

        #     # draw mean of posterior probability
        #     pyplot.figure(figsize=(6, 6))
        #     base = base / np.sum(base)
        #     # pyplot.pcolor(ratemap_free.X, ratemap_free.Y, base / (t_edge - l_edge), cmap=pyplot.cm.afmhot)
        #     pyplot.pcolor(ratemap_free.X, ratemap_free.Y, base, cmap=pyplot.cm.afmhot)
        #     pyplot.title("Prob max = %f" % np.max(base))
        #     pyplot.tick_params(labelbottom=False, bottom=False)
        #     pyplot.tick_params(labelleft=False, left=False)
        #     pyplot.box(False)
        #     pyplot.axis("equal")
        #     pyplot.clim([0, 0.08])
        #     pyplot.savefig(output_dir + r_trial_dir + "/mean.png")
        #     pyplot.close()
        #
        #     # draw x, y, mov
        #     fig = pyplot.figure(figsize=(6, 6))
        #     ax1 = fig.add_subplot(3, 1, 1)
        #     ax1.plot(event.t_vec, event.x, "-k")
        #     ax1.plot(event.t_vec, event.x, "or", ms=2.5)
        #     ax1.tick_params(labelbottom=False, bottom=False)
        #     ax1.set_ylim(ymin=0.0, ymax=params["env_size"][2])
        #     ax1.set_ylabel("X Pos [m]")
        #     # ax1.set_yticks([0.5, 1.0, 1.5])
        #     ax1.locator_params(axis='y', nbins=4)
        #     ax2 = fig.add_subplot(3, 1, 2)
        #     ax2.plot(event.t_vec, event.y, "-k")
        #     ax2.plot(event.t_vec, event.y, "or", ms=2.5)
        #     ax2.tick_params(labelbottom=False, bottom=False)
        #     ax2.set_ylim(ymin=0.0, ymax=params["env_size"][3])
        #     ax2.locator_params(axis='y', nbins=4)
        #     # ax2.set_yticks([0.05, 0.1, 0.15, 0.2])
        #     ax2.set_ylabel("Y Pos [m]")
        #     ax3 = fig.add_subplot(3, 1, 3)
        #     ax3.plot(event.t_vec[:-1], event.mov, "-k")
        #     ax3.plot(event.t_vec[:-1], event.mov, "or", ms=2.5)
        #     ax3.set_ylim(ymin=0, ymax=jump_lim)
        #     ax3.set_ylabel("Mov. [m/step]")
        #     ax3.set_xlabel("Time [sec]")
        #     pyplot.subplots_adjust(wspace=0.1, hspace=0.0)
        #     fig.savefig(output_dir + r_trial_dir + "/x_y_mov.png")
        #     # reverse replay
        #     if params["task"] == "linear_track" and event.x[0] - event.x[-1] > 0:
        #         fig.savefig("./reverse_replay/at%.3fto%.3fsec.png" % (event.t_vec[0], event.t_vec[-1]))
        #     pyplot.close(fig)
        #

    return event_list


def plot_sorted_raster(pyramidal_replay: sa.SpikeData, replay_event: TrajectoryEvent, x_y_hz: np.ndarray, time_window: list,
                       output_dir):
    r_trial_dir = output_dir + "/replay%.3fto%.3f" % (replay_event.t_vec[0], replay_event.t_vec[-1])
    sorted_id = x_y_hz[pyramidal_replay.cell_id - pyramidal_replay.member[0], 0]
    pyplot.plot(pyramidal_replay.spike_timing, sorted_id, "k.", ms=0.5, label="spikes")
    pyplot.fill_between([replay_event.t_vec[0], replay_event.t_vec[-1]], [1.5, 1.5], facecolor="r", alpha=0.1,
                        label="replay period")
    pyplot.xlim([time_window[0] - .25, time_window[1] + .25])
    pyplot.ylim([0, 1.5])
    pyplot.legend()
    pyplot.xlabel("Time [sec]")
    pyplot.ylabel("X pos. [m]")
    pyplot.savefig(r_trial_dir + "/sorted_raster.png")
    pyplot.close()


def main(root_dir: str, session_id: int, params: dict):

    # analysis parameters
    pc_lim = params["pc_lim"]
    sd_lim = params["ripple_lim"]
    jump_lim = params["jump_lim"]
    least_path = params["least_path"]
    least_step = params["least_step"]
    replay_simulation_time = params["replay_simulation_time"] * 0.001  # replay simulation time [sec]
    replay_analysis_start = params["replay_analysis_start"]  # analysis start time[sec]

    # -------- analysis of the hippocampal replay simulation -------- #
    for da_idx, da in enumerate(params["da_list"]):

        # make output directory
        r_trial_dir = root_dir + params["task"] + "_replay_fach%.1f/da%.2f/session%d" % (params["f_ach"], da, session_id)
        train_dir = root_dir + params["task"] + "_task/da%.2f/session%d" % (da, session_id)
        os.makedirs(r_trial_dir, exist_ok=True)

        # load network data
        with open(train_dir + "/snn_trial%d.pickle" % (params["n_train_trial"]-1), "rb") as f:
            snn: EcHpc = pickle.load(f)

        # calc PC rate map
        sr_free = sa.SpikeReader(np.load(train_dir + "/spike_total.npy"))
        tr_free = sa.TrajectoryReader(np.load(train_dir + "/place_total.npy"), env_size=params["env_size"])
        pyramidal_free = sr_free.make_group(snn.pyramidal.gid)
        traj_free = tr_free.make_trajectory()
        pyramidal_ratemap = sa.calc_ratemap2d_div(pyramidal_free, traj_free, time_window=[0, traj_free.t[-1]], space_bin_size=0.025, sigma=2)
        x_y_hz = sa.calc_center(pyramidal_ratemap)  # calc position of place field and peak firing rate

        # hippocampal replay detection from resting state simulation results
        for trial in range(params["n_replay_trial"]):
            print("\nAnalyzing da:%.2f stim_seed:%d" % (da, trial))
            # load data
            sr_replay = sa.SpikeReader(np.load(train_dir + "/replay_spike%d_fach%.1f.npy" % (trial, params["f_ach"])))
            pyramidal_replay = sr_replay.make_group(snn.pyramidal.gid)

            # find ripple segments
            lfp_pyramidal = sa.calc_group_firing_rate(pyramidal_replay,
                                                      time_window=[replay_analysis_start, replay_simulation_time],
                                                      time_bin_size=0.001)
            ripple_event = sa.find_ripple(lfp_pyramidal, sd_lim=sd_lim)

            # find replay from ripple segments
            replay_list, shuffled_replay_list = [], []
            for left, right in ripple_event.intervals:
                # exclude SWRs longer than 2sec or less than 50ms
                if (right - left >= 2.0) or (right - left < 0.05):
                    continue
                # do bayesian decoding analysis for the ripple segment
                event_list = replay_decoding(
                    output_dir=r_trial_dir + "/stim_seed%d" % trial,
                    ratemap_free=pyramidal_ratemap, replay_group=pyramidal_replay, x_y_hz=x_y_hz,
                    time_window=[left, right], pc_lim=pc_lim, least_path=least_path, jump_lim=jump_lim, least_step=least_step, params=params)
                # store replay event to the list
                for event in event_list:
                    if event.is_replay:
                        replay_list.append(event)

            # ---------------------------------- plot PC activity ---------------------------------- #
            def plot_swr_replay(min_val, max_val, ylabel):
                for left, right in ripple_event.intervals:
                    pyplot.fill_between([left, right], [max_val] * 2, facecolor="r", alpha=0.3)
                for re in replay_list:
                    pyplot.fill_between([re.t_vec[0], re.t_vec[-1]], [max_val] * 2, facecolor="b", alpha=0.3)
                pyplot.xlim([replay_analysis_start, replay_simulation_time])
                pyplot.ylim([min_val, max_val])
                pyplot.ylabel(ylabel)

            pyplot.figure(figsize=(10, 8))
            pyplot.subplot(411)
            pyplot.plot(pyramidal_replay.spike_timing, pyramidal_replay.cell_id, "k.", ms=0.5)
            plot_swr_replay(np.min(snn.pyramidal.gid), np.max(snn.pyramidal.gid), "Cell#")
            pyplot.subplot(412)
            pyplot.plot(lfp_pyramidal.t_vec, lfp_pyramidal.val, "k-")
            plot_swr_replay(0, np.max(lfp_pyramidal.val), "Mean Rate [Hz]")
            pyplot.subplot(413)
            pyplot.plot(ripple_event.filtered_rate.t_vec, ripple_event.filtered_rate.val, "k-")
            plot_swr_replay(0, np.max(ripple_event.filtered_rate.val), "Filtered Rate [Hz]")
            pyplot.subplot(414)
            pyplot.plot(ripple_event.envelope.t_vec, ripple_event.envelope.val, "k-")
            pyplot.plot([replay_analysis_start, replay_simulation_time],
                        [np.std(ripple_event.envelope.val) * sd_lim + np.mean(ripple_event.envelope.val)] * 2, "--k")
            pyplot.plot([replay_analysis_start, replay_simulation_time],
                        [np.mean(ripple_event.envelope.val)] * 2, "--k")
            plot_swr_replay(0, np.max(ripple_event.envelope.val), "Ripple Envelope")
            pyplot.xlabel("Time [sec]")
            pyplot.savefig(r_trial_dir + "/raster_stim_seed%d.png" % trial)
            pyplot.close()


if __name__ == "__main__":
    root_dir = sys.argv[1]
    session_id = int(sys.argv[2])
    # save parameter
    with open(root_dir + "/params.pickle", "rb") as f:
        params: dict = pickle.load(f)
    main(root_dir, session_id, params)
