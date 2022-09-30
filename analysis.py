import os
import numpy as np
import pickle
from ec_hpc_net import EcHpcSimple
from cusnn import analysis as sa
import pandas as pd
from scipy import stats


class TrajectoryEvent:
    def __init__(self, is_replay, t_vec, x, y, mov):
        self.is_replay = is_replay
        self.t_vec = t_vec
        self.x = x
        self.y = y
        self.mov = mov


def place_weight_analysis(root_dir, params: dict):

    # place-weight analysis parameter
    head, tail = -params["weight_width"] - 0.5 * params["space_bin_size"], params["weight_width"] - 0.5 * params["space_bin_size"]
    place_bins = np.arange(head, tail + params["space_bin_size"], params["space_bin_size"])

    # data array
    trial_change = np.zeros((params["n_train_trial"], place_bins.shape[0] - 1))
    da_change = np.zeros((len(params["da_list"]), place_bins.shape[0] - 1))

    # load data
    for da in params["da_list"]:
        for session in range(params["n_session"]):
            folder_name = root_dir + "linear_track_task/da%.2f/session%d" % (da, session)
            place_weight = np.load(folder_name + "/place_weight.npy")
            if da == params["da_fix"]:
                trial_change += place_weight[:params["n_train_trial"], :]
            da_change[params["da_list"].index(da)] += place_weight[params["trial_fix"] - 1]

    # calc average
    trial_change /= params["n_session"]
    da_change /= params["n_session"]

    # save data
    os.makedirs(root_dir + "/analysis", exist_ok=True)
    np.save(root_dir + "/analysis/trial_weight_change_1d.npy", trial_change)
    np.save(root_dir + "/analysis/da_weight_change_1d.npy", da_change)

    return trial_change, da_change


def calc_spike_skewness_x_axis(spikes: sa.SpikeData, trajectory: sa.TrajectoryData, time_window: list):
    skew_list, width_list, com_list = np.zeros(spikes.n_cells), np.zeros(spikes.n_cells), np.zeros(spikes.n_cells)
    for i in range(spikes.n_cells):
        # search cell's spike data
        index = np.where(spikes.cell_id == spikes.member[i])[0]
        cell_spike = spikes.spike_timing[index]  # ms
        cell_spike = cell_spike[(time_window[0] < cell_spike) & (cell_spike <= time_window[1])]
        # if fired at least once
        if cell_spike.size:
            x = trajectory.x[np.searchsorted(trajectory.t, cell_spike, side="right") - 1]
            skew_list[i] = stats.skew(x)
            width_list[i] = np.max(x) - np.min(x)
            com_list[i] = np.mean(x)
        else:
            skew_list[i] = np.nan
            width_list[i] = np.nan
            com_list[i] = np.nan
    return skew_list, width_list, com_list


def place_cell_analysis_1d(root_dir: str, params: dict):

    n_session = params["n_session"]
    n_trial = params["n_train_trial"]
    da_list = params["da_list"]
    out_dir = root_dir + "/analysis/"

    with open(root_dir + 'linear_track_task/da%.2f/session%d' % (da_list[0], 0) + "/snn_trial%d.pickle" % (n_trial - 1), "rb") as f:
        snn: EcHpcSimple = pickle.load(f)

    # parameters
    n_da = len(da_list)

    # place field analysis parameter
    field_thresh = 0.1  # [Hz]
    x_axis = np.arange(0.0, 1.5 + params["x_bin"], params["x_bin"])

    # output data array
    field_skew = np.zeros((n_da, n_trial, n_session))
    field_d_COM = np.zeros((n_da, n_trial, n_session))
    mean_COM_vs_d_COM = np.zeros((n_da, x_axis.size-1, n_session))
    mean_COM_vs_d_COM_freq = np.zeros((n_da, x_axis.size-1, n_session))

    # analysis loop
    for i, da in enumerate(da_list):
        for session in range(n_session):
            print("da: %.2f, session: %d" % (da, session))
            # load data
            train_dir = root_dir + 'linear_track_task/da%.2f/session%d' % (da, session)
            spike_tot_raw = np.load(train_dir + '/spike_total.npy')  # [timing, cell id]
            place_tot_raw = np.load(train_dir + "/place_total.npy")  # [t, x(t), y(t)]

            place_t_vec = (place_tot_raw[:, 0] * 1000).astype(np.int)
            spike_t_vec = (spike_tot_raw[:, 0] * 1000).astype(np.int)

            # make rate map for early and late trials
            t_sum = np.load(train_dir + "/trial_end_time.npy") + 1
            t_sum = np.cumsum(t_sum)
            t_sum = np.insert(t_sum, 0, 0)

            # remove initial act
            start_act = 2000  # [ms]
            place_tot = np.concatenate((place_tot_raw[:np.searchsorted(place_t_vec, 0)], place_tot_raw[np.searchsorted(place_t_vec, start_act):]))
            spike_tot = np.concatenate((spike_tot_raw[:np.searchsorted(spike_t_vec, 0)], spike_tot_raw[np.searchsorted(spike_t_vec, start_act):]))
            place_t_vec = np.concatenate((place_t_vec[:np.searchsorted(place_t_vec, 0)], place_t_vec[np.searchsorted(place_t_vec, start_act):]))
            spike_t_vec = np.concatenate((spike_t_vec[:np.searchsorted(spike_t_vec, 0)], spike_t_vec[np.searchsorted(spike_t_vec, start_act):]))

            # stack all trial data
            for j in range(1, n_trial):
                t_s, t_as = t_sum[j], t_sum[j] + start_act
                place_tot = np.concatenate((place_tot[:np.searchsorted(place_t_vec, t_s)], place_tot[np.searchsorted(place_t_vec, t_as):]))
                spike_tot = np.concatenate((spike_tot[:np.searchsorted(spike_t_vec, t_s)], spike_tot[np.searchsorted(spike_t_vec, t_as):]))
                place_t_vec = np.concatenate((place_t_vec[:np.searchsorted(place_t_vec, t_s)], place_t_vec[np.searchsorted(place_t_vec, t_as):]))
                spike_t_vec = np.concatenate((spike_t_vec[:np.searchsorted(spike_t_vec, t_s)], spike_t_vec[np.searchsorted(spike_t_vec, t_as):]))

            p_edge = [np.searchsorted(place_t_vec, t_sum[j]) for j in range(n_trial + 1)]
            s_edge = [np.searchsorted(spike_t_vec, t_sum[j]) for j in range(n_trial + 1)]

            # data array
            rate_maps, pyr_spk_list, traj_list = [], [], []
            n_fields = []

            # make rate maps
            for j in range(n_trial):
                tr = sa.TrajectoryReader(place_tot[p_edge[j]:p_edge[j+1]], env_size=[0, 0, 1.5, 0.2])
                sr = sa.SpikeReader(spike_tot[s_edge[j]:s_edge[j+1]])
                trajectory = tr.make_trajectory()
                pyr = sr.make_group(snn.pyramidal.gid)
                pyr_spk_list.append(pyr)
                traj_list.append(trajectory)
                rate_maps.append(sa.calc_ratemap2d(pyr, trajectory, time_window=[0, 1000000]))

            # nan -> 0 for this analysis
            for r_map in rate_maps:
                r_map.rate[np.isnan(r_map.rate)] = 0
            # find place cells
            pc_idx = np.ones(rate_maps[0].n_cells, dtype=np.bool)
            for j in range(n_trial):
                n_fields.append(sa.calc_place_field_number(rate_maps[j], thresh_rate=field_thresh))
                peaks = np.max(np.max(rate_maps[j].rate, axis=1), axis=1)
                pc_idx &= (peaks > params["pc_lim"]) & (n_fields[j] == 1)
            # make mean map
            mean_rate_map = rate_maps[0].rate * 0.0
            for j in range(n_trial):
                mean_rate_map += rate_maps[j].rate / n_trial
            # calc mean COM
            sr = sa.SpikeReader(spike_tot)
            tr = sa.TrajectoryReader(place_tot, env_size=[0, 0, 1.5, 0.2])
            trajectory = tr.make_trajectory()
            pyr = sr.make_group(snn.pyramidal.gid)
            tot_ratemap = sa.calc_ratemap2d(pyr, trajectory, time_window=[0, 1000000])
            tot_ratemap.rate[np.isnan(tot_ratemap.rate)] = 0    # nan -> 0
            tot_skew, tot_width, tot_COM = calc_spike_skewness_x_axis(pyr, trajectory, time_window=[0, 1000000])

            # calc in-field firing rate, peak firing rate, spatial coherence
            from scipy import signal
            coherence_kernel = np.ones((3, 3)) / 8
            coherence_kernel[1, 1] = 0
            in_field_rate = np.zeros(pyr.n_cells)
            peak_firing_rate = np.zeros(pyr.n_cells)
            spatial_coherence = np.zeros(pyr.n_cells)
            peak_x_pos = np.zeros(pyr.n_cells)
            pf_size = np.zeros(pyr.n_cells)
            for j in range(pyr.n_cells):
                rate_map = tot_ratemap.rate[j]
                peak_firing_rate[j] = np.max(rate_map)
                spatial_coherence[j], _ = stats.pearsonr(rate_map[1:-1, 1:-1].flatten(), signal.convolve2d(rate_map,
                                                                                  coherence_kernel, "same")[1:-1, 1:-1].flatten())
                if np.sum(rate_map) == 0:
                    peak_x_pos[j] = np.nan
                    in_field_rate[j] = np.nan
                    pf_size[j] = np.nan
                else:
                    peak_x_pos[j] = np.argmax(np.sum(rate_map, axis=0)) * tot_ratemap.bin_size
                    in_field_rate[j] = np.mean(rate_map[rate_map > peak_firing_rate[j] * 0.1])  # 10% of peak
                    wid = np.where(np.sum(rate_map > peak_firing_rate[j] * 0.1, axis=0) != 0)[0]
                    pf_size[j] = (1 + wid[-1] - wid[0]) * tot_ratemap.bin_size

            # trial-by-trial analysis
            for j in range(n_trial):
                skew, width, com = calc_spike_skewness_x_axis(pyr_spk_list[j], traj_list[j], time_window=[0, 1000000])
                field_skew[i, j, session] = np.mean(skew[pc_idx])
                d_COM = np.zeros(rate_maps[0].n_cells)
                for k in np.where(pc_idx == np.True_)[0]:
                    d_COM[k] = com[k] - tot_COM[k]
                    if j == 0:
                        x_idx = np.searchsorted(x_axis, tot_COM[k])
                        mean_COM_vs_d_COM[i, x_idx, session] -= d_COM[k]
                        mean_COM_vs_d_COM_freq[i, x_idx, session] += 1
                    elif j == n_trial-1:
                        x_idx = np.searchsorted(x_axis, tot_COM[k])
                        mean_COM_vs_d_COM[i, x_idx, session] += d_COM[k]
                field_d_COM[i, j, session] = np.mean(d_COM[pc_idx])

            # save place cell property as DataFrame
            place_cell_stats = pd.DataFrame({"Cell#": snn.pyramidal.gid,
                                             "Skewness": tot_skew,
                                             "Spatial Information": sa.calc_spatial_information(tot_ratemap),
                                             "Number of Fields": sa.calc_place_field_number(tot_ratemap, thresh_rate=field_thresh),
                                             "Field Width": tot_width,
                                             "Place Field Size": pf_size,
                                             "In-Field Firing Rate": in_field_rate,
                                             "Peak Firing Rate": peak_firing_rate,
                                             "Spatial Coherence": spatial_coherence,
                                             "Peak Pos X": peak_x_pos
                                             })
            place_cell_stats = place_cell_stats.sort_values(by="Cell#")
            place_cell_stats.to_pickle(train_dir + "/place_cell_stats.pickle")

    mean_COM_vs_d_COM[np.nonzero(mean_COM_vs_d_COM_freq)] /= mean_COM_vs_d_COM_freq[np.nonzero(mean_COM_vs_d_COM_freq)]

    # save data
    np.save(out_dir + "field_skew_1d.npy", field_skew)
    np.save(out_dir + "field_d_COM_1d.npy", field_d_COM)
    np.save(out_dir + "mean_COM_vs_d_COM_1d.npy", mean_COM_vs_d_COM)


def place_cell_analysis_2d(root_dir: str, params: dict):

    out_dir = root_dir + "analysis_2d/"
    os.makedirs(out_dir, exist_ok=True)
    with open(root_dir + 'water_maze_task/da%.2f/session%d' % (params["da_list"][0], 0) + "/snn_trial%d.pickle" %
              (params["n_train_trial"] - 1), "rb") as f:
        snn: EcHpcSimple = pickle.load(f)

    # parameters
    n_da = len(params["da_list"])
    session_div = params["session_div"]

    # place field analysis parameter
    field_thresh = 0.1  # [Hz]

    # axis
    theta_axis = np.linspace(0, 2 * np.pi, params["n_part"])
    mov_axis = np.linspace(0, params["mov_lim"], params["n_part"])

    # output data array
    theta_v_dist = np.zeros((n_da, session_div, session_div, theta_axis.size - 1, mov_axis.size))
    field_area = np.zeros((n_da, session_div, params["n_session"]))
    field_peak = np.zeros((n_da, session_div, params["n_session"]))
    pyr_mean_rate = np.zeros((n_da, session_div, params["n_session"]))

    # analysis loop
    for i, da in enumerate(params["da_list"]):
        mov_list = []
        theta_list = []
        for session in range(params["n_session"]):
            print("da: %.2f, session: %d" % (da, session))
            # load data
            train_dir = root_dir + 'water_maze_task/da%.2f/session%d' % (da, session)

            spike_0 = np.load(train_dir + '/spike_trial0.npy')  # [timing, cell id]
            place_0 = np.load(train_dir + '/place_trial0.npy')  # [t, x(t), y(t)]
            spike_35 = np.load(train_dir + '/spike_trial35.npy')  # [timing, cell id]
            place_35 = np.load(train_dir + '/place_trial35.npy')  # [t, x(t), y(t)]

            tr = sa.TrajectoryReader(place_0, env_size=[0, 0, 1.5, 1.5])
            sr = sa.SpikeReader(spike_0)
            trajectory = tr.make_trajectory()
            pyr = sr.make_group(snn.pyramidal.gid)
            rate_map_0 = sa.calc_ratemap2d_div(pyr, trajectory, time_window=[params["act_start"] * 0.001, trajectory.t[-1]], no_nan=True)

            tr = sa.TrajectoryReader(place_35, env_size=[0, 0, 1.5, 1.5])
            sr = sa.SpikeReader(spike_35)
            trajectory = tr.make_trajectory()
            pyr = sr.make_group(snn.pyramidal.gid)
            rate_map_35 = sa.calc_ratemap2d_div(pyr, trajectory, time_window=[params["act_start"] * 0.001, trajectory.t[-1]], no_nan=True)

            n_fields_0 = sa.calc_place_field_number(rate_map_0, thresh_rate=field_thresh)
            n_fields_35 = sa.calc_place_field_number(rate_map_35, thresh_rate=field_thresh)

            # pair wise analysis
            a_max = np.nanmax(np.nanmax(rate_map_0.rate, axis=1), axis=1) > params["pc_lim"]
            b_max = np.nanmax(np.nanmax(rate_map_35.rate, axis=1), axis=1) > params["pc_lim"]
            pc_idx = np.where(((a_max & b_max) == np.True_) & (n_fields_0 == 1) & (n_fields_35 == 1))[0]

            # calc map correlation and vector distribution
            for ii, idx in enumerate(pc_idx):
                e_sum, l_sum = np.sum(rate_map_0.rate[idx]), np.sum(rate_map_35.rate[idx])
                a_y, a_x = np.sum(rate_map_0.rate[idx] * rate_map_0.Y) / e_sum, np.sum(rate_map_0.rate[idx] * rate_map_0.X) / e_sum
                b_y, b_x = np.sum(rate_map_35.rate[idx] * rate_map_35.Y) / l_sum, np.sum(rate_map_35.rate[idx] * rate_map_35.X) / l_sum
                mov = np.sqrt((b_y - a_y) ** 2 + (b_x - a_x) ** 2)  # [m]
                theta = np.arctan2(b_y - a_y, b_x - a_x)  # -pi~pi[rad]
                theta = np.mod(theta + np.pi * 2, np.pi * 2)  # 0~2pi[rad]
                mov_list.append(mov)

                if 0.05 < mov <= params["mov_lim"]:
                    theta_list.append(theta)
                    theta_v_dist[i, 0, 1, :, np.searchsorted(mov_axis, mov, "right") - 1] += np.histogram(theta, bins=theta_axis)[0]

        np.save(out_dir + "mov_sample_da%.1f.npy" % da, np.array(mov_list))
        np.save(out_dir + "theta_sample_da%.1f.npy" % da, np.array(theta_list))

    # save data
    np.save(out_dir + "theta_v_dist.npy", theta_v_dist)
    np.save(out_dir + "field_area.npy", field_area)
    np.save(out_dir + "field_peak.npy", field_peak)
    np.save(out_dir + "pyr_mean_rate.npy", pyr_mean_rate)


def replay_stats_1d(root_dir: str, params: dict):

    # analysis parameters
    n_da = len(params["da_list"])

    # data array
    replay_freq = np.zeros((n_da, params["n_session"]))

    st_end = np.zeros((n_da, params["n_session"]))
    mov_tot = np.zeros((n_da, params["n_session"]))
    mov_skew = np.zeros((n_da, params["n_session"]))
    mean_x = np.zeros((n_da, params["n_session"]))
    velocity = np.zeros((n_da, params["n_session"]))

    velocity_mu = np.zeros(n_da)
    velocity_se = np.zeros(n_da)
    st_end_mu = np.zeros(n_da)
    st_end_se = np.zeros(n_da)

    # -------- analysis of the hippocampal replay simulation -------- #
    for da_idx, da in enumerate(params["da_list"]):

        da_velocity = []
        da_st_end = []

        start_pos = []
        end_pos = []

        for session in range(params["n_session"]):
            # session data
            ses_st_end = []
            ses_mov_tot = []
            ses_mov_skew = []
            ses_mean_x = []
            ses_velocity = []

            # -------- analysis of the free exploration simulation -------- #
            r_trial_dir = root_dir + "linear_track_replay_fach%.1f/da%.2f/session%d/" % (params["f_ach"], da, session)
            session_files = os.listdir(r_trial_dir)
            stim_dirs = [f for f in session_files if os.path.isdir(os.path.join(r_trial_dir, f))]
            for stim_dir in stim_dirs:
                # load data
                path = r_trial_dir + stim_dir
                stim_files = os.listdir(path)

                replay_files = [f for f in stim_files if os.path.isfile(os.path.join(path, f)) and ("pickle" in f)]

                for replay_file_name in replay_files:
                    with open(path + "/" + replay_file_name, "rb") as f:
                        re: TrajectoryEvent = pickle.load(f)

                    # store data
                    replay_freq[da_idx, session] += 1
                    ses_st_end.append(re.x[-1] - re.x[0])
                    ses_mov_tot.append(np.sum(re.mov))
                    ses_mov_skew.append(stats.skew(re.mov))
                    ses_mean_x.append(np.mean(re.x))
                    ses_velocity.append(np.sum(re.mov) / (0.005 * re.mov.size))
                    da_velocity.append(np.sum(re.mov) / (0.005 * re.mov.size))
                    da_st_end.append(re.x[-1] - re.x[0])

                    start_pos.append(re.x[0])
                    end_pos.append(re.x[-1])

            st_end[da_idx, session] = np.mean(ses_st_end)
            mov_tot[da_idx, session] = np.mean(ses_mov_tot)
            mov_skew[da_idx, session] = np.mean(ses_mov_skew)
            mean_x[da_idx, session] = np.mean(ses_mean_x)
            velocity[da_idx, session] = np.mean(ses_velocity)

        velocity_mu[da_idx] = np.mean(da_velocity)
        velocity_se[da_idx] = np.std(da_velocity) / np.sqrt(len(da_velocity))
        st_end_mu[da_idx] = np.mean(da_st_end)
        st_end_se[da_idx] = np.std(da_st_end) / np.sqrt(len(da_st_end))

    replay_freq = replay_freq / (params["replay_simulation_time"]*0.001 - params["replay_analysis_start"]) / (params["n_replay_trial"])

    # save data
    out_dir = root_dir + "/analysis/f_ach%.1f/" % params["f_ach"]
    os.makedirs(root_dir + "/analysis/f_ach%.1f" % params["f_ach"], exist_ok=True)
    np.save(out_dir + "replay_freq_1d.npy", replay_freq)
    np.save(out_dir + "st_end_1d.npy", st_end)
    np.save(out_dir + "mov_tot_1d.npy", mov_tot)
    np.save(out_dir + "mov_skew_1d.npy", mov_skew)
    np.save(out_dir + "replay_mean_x_1d.npy", mean_x)
    np.save(out_dir + "velocity_mean_1d.npy", velocity)

    np.save(out_dir + "velocity_mu_1d.npy", velocity_mu)
    np.save(out_dir + "velocity_se_1d.npy", velocity_se)
    np.save(out_dir + "st_end_mu_1d.npy", st_end_mu)
    np.save(out_dir + "st_end_se_1d.npy", st_end_se)


def replay_stats_2d(root_dir: str, params: dict):

    out_dir = root_dir + "/analysis_2d/f_ach%.1f/" % params["f_ach"]
    os.makedirs(out_dir, exist_ok=True)

    # analysis parameters
    n_da = len(params["da_list"])

    # data array
    ripple_freq = np.zeros((n_da, params["n_session"]))
    replay_freq = np.zeros((n_da, params["n_session"]))

    mov_tot = np.zeros((n_da, params["n_session"]))
    mov_skew = np.zeros((n_da, params["n_session"]))
    velocity = np.zeros((n_da, params["n_session"]))

    start_pos = []
    end_pos = []

    # -------- analysis of the hippocampal replay simulation -------- #
    for session in range(params["n_session"]):
        print("session #%d" % session)
        for da_idx, da in enumerate(params["da_list"]):
            # session data
            ses_mov_tot = []
            ses_mov_skew = []
            ses_velocity = []

            # -------- analysis of the free exploration simulation -------- #
            r_trial_dir = root_dir + "water_maze_replay_fach%.1f/da%.2f/session%d/" % (params["f_ach"], da, session)
            session_files = os.listdir(r_trial_dir)
            stim_dirs = [f for f in session_files if os.path.isdir(os.path.join(r_trial_dir, f))]
            for stim_dir in stim_dirs:
                # load data
                path = r_trial_dir + stim_dir
                stim_files = os.listdir(path)
                replay_files = [f for f in stim_files if os.path.isfile(os.path.join(path, f)) and ("pickle" in f)]

                for replay_file_name in replay_files:
                    with open(path + "/" + replay_file_name, "rb") as f:
                        re: TrajectoryEvent = pickle.load(f)
                    if re.is_replay is True:
                        # store data
                        replay_freq[da_idx, session] += 1
                        ses_mov_tot.append(np.sum(re.mov))
                        ses_mov_skew.append(stats.skew(re.mov))
                        ses_velocity.append(np.sum(re.mov) / (0.005 * re.mov.size))
                        start_pos.append(np.array([re.x[0], re.y[0]]))
                        end_pos.append(np.array([re.x[-1], re.y[-1]]))

            mov_tot[da_idx, session] = np.mean(ses_mov_tot)
            mov_skew[da_idx, session] = np.mean(ses_mov_skew)
            velocity[da_idx, session] = np.mean(ses_velocity)

    replay_freq = replay_freq / (params["replay_simulation_time"]*0.001 - params["replay_analysis_start"]) / (params["n_replay_trial"])
    ripple_freq = ripple_freq / (params["replay_simulation_time"]*0.001 - params["replay_analysis_start"]) / (params["n_replay_trial"])

    # save data
    np.save(out_dir + "replay_freq_2d.npy", replay_freq)
    np.save(out_dir + "ripple_freq_2d.npy", ripple_freq)
    np.save(out_dir + "mov_tot_2d.npy", mov_tot)
    np.save(out_dir + "mov_skew_2d.npy", mov_skew)
    np.save(out_dir + "velocity_mean_2d.npy", velocity)
    np.save(out_dir + "start_pos.npy", np.stack(start_pos))
    np.save(out_dir + "end_pos.npy", np.stack(end_pos))

