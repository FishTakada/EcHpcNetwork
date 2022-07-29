import numpy as np
import pickle
from matplotlib import pyplot
import matplotlib.cm as cm
from matplotlib import mathtext
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from seaborn import heatmap
from scipy import stats
from cusnn import analysis as sa
from scipy.stats import ttest_ind
import pandas as pd


def set_text():
    mathtext.FontConstantsBase = mathtext.ComputerModernFontConstants
    pyplot.rcParams['font.family'] = 'Arial'
    pyplot.rcParams['xtick.direction'] = 'in'
    pyplot.rcParams['ytick.direction'] = 'in'
    pyplot.rcParams['xtick.major.width'] = 1.0
    pyplot.rcParams['ytick.major.width'] = 1.0
    pyplot.rcParams['font.size'] = 8
    pyplot.rcParams['axes.linewidth'] = 1.0
    pyplot.rcParams['lines.markersize'] = 0
    pyplot.rcParams['errorbar.capsize'] = 0


def cm2inch(value):
    return value/2.54


def figure1(data_dir):

    # load ratemap
    with open(data_dir + "/grid_1_ratemap.pickle", "rb") as f:
        g1_ratemap: sa.RateMap = pickle.load(f)
    with open(data_dir + "/grid_2_ratemap.pickle", "rb") as f:
        g2_ratemap: sa.RateMap = pickle.load(f)
    with open(data_dir + "/pyramidal_ratemap.pickle", "rb") as f:
        pyr_ratemap: sa.RateMap = pickle.load(f)

    # figure for paper
    fig = pyplot.figure(figsize=(cm2inch(15.0), cm2inch(6.0)))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title("GC1")
    mappable = ax1.pcolor(g1_ratemap.X, g1_ratemap.Y, g1_ratemap.rate[17], cmap="jet", edgecolors="none")
    ax1.tick_params(labelbottom=False, bottom=False)
    ax1.tick_params(labelleft=False, left=False)
    ax1.set_xticklabels([])
    ax1.axis('equal')
    fig.colorbar(mappable, label="Firing Rate [Hz]", orientation="horizontal", shrink=0.6, pad=0.1)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title("GC2")
    mappable = ax2.pcolor(g2_ratemap.X, g2_ratemap.Y, g2_ratemap.rate[23], cmap="jet", edgecolors="none")
    ax2.tick_params(labelbottom=False, bottom=False)
    ax2.tick_params(labelleft=False, left=False)
    ax2.set_xticklabels([])
    ax2.axis('equal')
    fig.colorbar(mappable, label="Firing Rate [Hz]", orientation="horizontal", shrink=0.6, pad=0.1)

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_title("PC")
    mappable = ax3.pcolor(pyr_ratemap.X, pyr_ratemap.Y, pyr_ratemap.rate[21], cmap="jet", edgecolors="none")
    ax3.tick_params(labelbottom=False, bottom=False)
    ax3.tick_params(labelleft=False, left=False)
    ax3.set_xticklabels([])

    ax3.axis('equal')
    fig.colorbar(mappable, label="Firing Rate [Hz]", orientation="horizontal", shrink=0.6, pad=0.1)

    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    ax3.spines['left'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    pyplot.tight_layout()
    pyplot.savefig(data_dir + "figure1_receptive_fields.eps", transparent=True, dpi=300)
    pyplot.close()


def paper_plot_1d(root_dir, params: dict):

    set_text()

    analysis_dir = root_dir + "/analysis/"
    fig_dir = root_dir + "/figures/"
    da_list = params["da_list"]
    n_da = len(da_list)

    # ---------------------------------------- figure 3 ---------------------------------------- #
    mean_se_peak_firing_rate = []
    mean_se_in_field_rate = []
    mean_se_sp_info = []
    mean_se_skewness = []
    mean_se_pf_size = []

    # analysis loop
    for i, da in enumerate(da_list):
        df_concat = None
        for session in range(params["n_session"]):
            # load data
            train_dir = root_dir + 'linear_track_task/da%.2f/session%d' % (da, session)
            place_cell_stats = pd.read_pickle(train_dir + "/place_cell_stats.pickle")
            df_concat = pd.concat([df_concat, place_cell_stats])

        pc_idx = df_concat["Number of Fields"] == 1
        root_n = np.sqrt(np.sum(pc_idx))
        mean_se_peak_firing_rate.append([np.mean(df_concat["Peak Firing Rate"][pc_idx]), np.std(df_concat["Peak Firing Rate"][pc_idx])/root_n])
        mean_se_in_field_rate.append([np.mean(df_concat["In-Field Firing Rate"][pc_idx]), np.std(df_concat["In-Field Firing Rate"][pc_idx])/root_n])
        mean_se_sp_info.append([np.mean(df_concat["Spatial Information"][pc_idx]), np.std(df_concat["Spatial Information"][pc_idx])/root_n])
        mean_se_skewness.append([np.mean(df_concat["Skewness"][pc_idx]), np.std(df_concat["Skewness"][pc_idx])/root_n])
        mean_se_pf_size.append([np.mean(df_concat["Place Field Size"][pc_idx]), np.std(df_concat["Place Field Size"][pc_idx])/root_n])

    fig = pyplot.figure(figsize=(cm2inch(19.0), cm2inch(13.0)))
    fig.subplots_adjust(wspace=0.4, hspace=0.25, left=0.1, right=0.95, bottom=0.1, top=0.95)

    def plot_mean(index, data, name):
        ax = fig.add_subplot(2, 3, index)
        y, se = [], []
        for i_da in range(n_da):
            y.append(data[i_da][0])
            se.append(data[i_da][1])
        pyplot.errorbar(da_list, y, yerr=se, fmt='o-k')
        ax.set_ylabel(name)
        ax.set_xlabel("$R_{DA}$")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.locator_params(axis='y', nbins=6)
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    plot_mean(5, mean_se_pf_size, "Place field size [m]")
    plot_mean(3, mean_se_peak_firing_rate, "Peak firing rate [Hz]")
    plot_mean(2, mean_se_in_field_rate, "In-field firing rate [Hz]")
    plot_mean(4, mean_se_sp_info, "Spatial information [bit/spike]")
    plot_mean(6, mean_se_skewness, "Skewness")
    pyplot.savefig(fig_dir + "figure3_place_cell_stats.png", transparent=True, dpi=300)
    pyplot.savefig(fig_dir + "figure3_place_cell_stats.eps", transparent=True, dpi=300)
    pyplot.close(fig)

    # ---------------------------------------- figure 4 ---------------------------------------- #
    wid = params["weight_width"]
    head, tail = -wid - 0.5 * params["space_bin_size"], wid - 0.5 * params["space_bin_size"]
    place_bins = np.arange(head, tail + params["space_bin_size"], params["space_bin_size"])

    # data array
    trial_change = np.load(analysis_dir + "trial_weight_change_1d.npy")
    da_change = np.load(analysis_dir + "da_weight_change_1d.npy")

    # plot place weight relation
    fig = pyplot.figure(figsize=(cm2inch(15.0), cm2inch(15.0)))
    fig.subplots_adjust(wspace=0.3, hspace=0.25, left=0.05, right=0.95, bottom=0.1, top=0.95)
    ax1 = fig.add_subplot(2, 2, 1)
    for i in range(0, params["n_train_trial"]):
        cval = i / (params["n_train_trial"]-1) * 0.85 + 0.075
        ax1.plot(place_bins[:-1] + 0.5 * params["space_bin_size"], trial_change[i], "-",
                 color=cm.jet(cval), label="%d" % (i+1))
    ax1.plot([0, 0], [-0.01, 0.2], "--k")
    ax1.legend(bbox_to_anchor=(0.02, 1.1), loc="upper left", framealpha=0.0, title="Trial#", labelspacing=0.3)
    ax1.set_xlabel("Distance between place fields [m]")
    ax1.set_ylabel("Mean synaptic weight")
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.set_ylim(-0.01, 0.2)
    ax1.set_xlim(-wid, wid)
    ax1.locator_params(axis='y', nbins=6)

    box = ax1.get_position()
    box.x0 = box.x0 + 0.05
    box.x1 = box.x1 + 0.05
    ax1.set_position(box)

    ax2 = fig.add_subplot(2, 2, 2)
    for i in range(0, n_da):
        cval = i / (n_da-1) * 0.85 + 0.075
        ax2.plot(place_bins[:-1] + 0.5 * params["space_bin_size"], da_change[i], "-",
                 color=cm.jet(cval), label="%.1f" % da_list[i])
    ax2.plot([0, 0], [-0.01, 0.2], "--k")
    ax2.legend(bbox_to_anchor=(0.02, 1.1), loc="upper left", framealpha=0.0, title="$R_{DA}$", labelspacing=0.3)
    ax2.set_xlabel("Distance between place fields [m]")
    ax2.tick_params(labelleft=False, left=False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.yaxis.set_ticks_position('left')
    ax2.xaxis.set_ticks_position('bottom')
    ax2.set_xlim(-wid, wid)
    ax2.set_ylim(-0.01, 0.2)
    ax2.locator_params(axis='y', nbins=6)

    box = ax2.get_position()
    box.x0 = box.x0 + 0.03
    box.x1 = box.x1 + 0.03
    ax2.set_position(box)

    pyplot.savefig(fig_dir + "figure4_place_weight_trial.eps", transparent=True, dpi=300)
    pyplot.savefig(fig_dir + "figure4_place_weight_trial.png", transparent=True, dpi=300)
    pyplot.close()

    # ---------------------------------------- figure 5 ---------------------------------------- #
    x = np.arange(1, params["n_train_trial"] + 1)
    x_axis = np.arange(0.0, 1.5 + params["x_bin"], params["x_bin"])

    # save data
    field_skew = np.load(analysis_dir + "field_skew_1d.npy")
    field_d_COM = np.load(analysis_dir + "field_d_COM_1d.npy") * 100
    mean_COM_vs_d_COM = np.load(analysis_dir + "mean_COM_vs_d_COM_1d.npy") * 100

    data_list = [field_d_COM, field_skew]
    name_list = ["ΔCOM [cm]", "Mean field skewness"]
    fig = pyplot.figure(figsize=(cm2inch(19.0), cm2inch(13.0)))
    fig.subplots_adjust(wspace=0.4, hspace=0.25, left=0.1, right=0.95, bottom=0.1, top=0.95)
    cnt = 1
    for data, name in zip(data_list, name_list):
        ax1 = fig.add_subplot(2, 3, cnt+2)
        for i_da in range(n_da):
            y = np.mean(data[i_da], axis=1)
            y_err = np.std(data[i_da], axis=1) / np.sqrt(params["n_session"])
            pyplot.errorbar(x, y, yerr=y_err, fmt='o-', color=cm.jet(i_da / (n_da-1)), label="%.1f" % da_list[i_da])
        ax1.set_ylabel(name)
        ax1.set_xlabel("Trial#")
        ax1.set_xlim(1, params["n_train_trial"]+0.5)
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.yaxis.set_ticks_position('left')
        ax1.xaxis.set_ticks_position('bottom')
        ax1.locator_params(axis='y', nbins=6)
        cnt += 1

    ax1 = fig.add_subplot(2, 3, 5)
    ax1.clear()
    first, last = np.mean(field_d_COM[:, 0, :], axis=1), np.mean(field_d_COM[:, params["n_train_trial"]-1, :], axis=1)
    ax1.errorbar(da_list, first-last, fmt='o-', color="r")
    ax1.set_ylabel("Total ΔCOM [cm]")
    ax1.set_xlabel("$R_{DA}$")
    ax1.yaxis.label.set_color('r')
    ax1.tick_params(axis='y', colors='red')
    ax1.set_xlim(min(da_list), max(da_list))
    ax2 = ax1.twinx()
    ax2.errorbar(da_list, np.mean(field_skew[:, params["n_train_trial"]-1, :], axis=1), fmt='o-', color="b")
    ax2.set_ylabel("Mean skewness (Trial#10)")
    ax2.yaxis.label.set_color('b')
    ax2.spines['right'].set_color('b')
    ax2.spines['left'].set_color('red')
    ax2.tick_params(axis='y', colors='b')

    box = ax1.get_position()
    box.x0 = box.x0 - 0.03
    box.x1 = box.x1 - 0.03
    ax1.set_position(box)
    ax2.set_position(box)

    ax1 = fig.add_subplot(2, 3, 6)
    for i_da in range(n_da):
        y = np.mean(mean_COM_vs_d_COM[i_da], axis=1)
        y_err = np.std(mean_COM_vs_d_COM[i_da], axis=1) / np.sqrt(params["n_session"])
        pyplot.errorbar(x_axis[:-1]*100, y, yerr=y_err, fmt='o-', color=cm.jet(i_da / (n_da - 1)), label="%.1f" % da_list[i_da])
    ax1.set_ylabel("ΔCOM [cm]")
    ax1.set_xlabel("Location [cm]")
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.locator_params(axis='y', nbins=6)

    pyplot.savefig(fig_dir + "figure5_place_field_stats_1d.eps", transparent=True, dpi=300)
    pyplot.savefig(fig_dir + "figure5_place_field_stats_1d.png", transparent=True, dpi=300)
    pyplot.close()

    # ---------------------------------------- figure 6 ---------------------------------------- #
    ach_list = [1.0, 1.5, 2.0, 2.5, 3.0]
    n_ach = len(ach_list)

    fig = pyplot.figure(figsize=(cm2inch(19.0), cm2inch(13.0)))
    fig.subplots_adjust(wspace=0.4, hspace=0.25, left=0.1, right=0.95, bottom=0.1, top=0.95)
    ax1 = fig.add_subplot(2, 3, 3)
    for i, ach in enumerate(ach_list):
        replay_freq = np.load(analysis_dir + "f_ach%.1f/replay_freq_1d.npy" % ach)
        y = np.nanmean(replay_freq, axis=1)
        y_err = np.nanstd(replay_freq, axis=1) / np.sqrt(params["n_session"])
        ax1.errorbar(da_list, y, yerr=y_err, fmt='o-', color=cm.jet(i / (n_ach - 1)), label="%.1f" % ach)
    ax1.set_xlim(0.0, 1.0)
    ax1.set_xlabel("$R_{DA}$")
    ax1.set_ylabel("Replay frequency [1/s]")
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.locator_params(axis='y', nbins=6)
    ax1.legend(framealpha=0.0, title="$f_{ACh}$")

    ax2 = fig.add_subplot(2, 3, 5)
    for i, ach in enumerate(ach_list):
        if ach < 2.0:
            continue
        velocity_mu = np.load(analysis_dir + "f_ach%.1f/velocity_mu_1d.npy" % ach)
        velocity_se = np.load(analysis_dir + "f_ach%.1f/velocity_se_1d.npy" % ach)
        ax2.errorbar(da_list, velocity_mu, yerr=velocity_se, fmt='o-', color=cm.jet(i / (n_ach - 1)), label="%.1f" % ach)
    ax2.set_xlim(0.0, 1.0)
    ax2.set_xlabel("$R_{DA}$")
    ax2.set_ylabel("Replay velocity [m/s]")
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.yaxis.set_ticks_position('left')
    ax2.xaxis.set_ticks_position('bottom')
    ax2.locator_params(axis='y', nbins=6)

    da_peaks = np.max(da_change, axis=1)
    half_width = np.zeros(n_da)
    for i, x_series in enumerate(da_change):
        for j in range(1, x_series.size):
            if x_series[j - 1] < da_peaks[i] * 0.5 <= x_series[j]:
                half_width[i] -= j
            if x_series[j - 1] > da_peaks[i] * 0.5 >= x_series[j]:
                half_width[i] += j
                break
    half_width *= params["space_bin_size"]
    ax4 = fig.add_subplot(2, 3, 4)
    pk_hw_rf = np.zeros((n_da * len(ach_list), 3))
    for i, ach in enumerate(ach_list):
        replay_freq = np.load(analysis_dir + "f_ach%.1f/replay_freq_1d.npy" % ach)
        pk_hw_rf[i * n_da:(i + 1) * n_da, 0] = da_peaks * ach
        pk_hw_rf[i * n_da:(i + 1) * n_da, 1] = half_width
        pk_hw_rf[i * n_da:(i + 1) * n_da, 2] = np.nanmean(replay_freq, axis=1)
    mappable = ax4.scatter(pk_hw_rf[:, 0], pk_hw_rf[:, 1], c=pk_hw_rf[:, 2], vmin=0, vmax=np.max(pk_hw_rf[:, 2]), s=35, cmap=cm.jet)
    ax4.set_xlabel("Peak PC to PC weight")
    ax4.set_ylabel("Half width [m]")
    cbaxes = inset_axes(ax4, width="60%", height="5%", loc="lower right", borderpad=2)
    fig.colorbar(mappable=mappable, cax=cbaxes, orientation='horizontal')

    ax3 = fig.add_subplot(2, 3, 6)
    for i, ach in enumerate(ach_list):
        if ach < 2.0:
            continue
        st_end_mu = np.load(analysis_dir + "f_ach%.1f/st_end_mu_1d.npy" % ach)
        st_end_se = np.load(analysis_dir + "f_ach%.1f/st_end_se_1d.npy" % ach)
        ax3.errorbar(da_list, st_end_mu, yerr=st_end_se, fmt='o-', color=cm.jet(i / (n_ach - 1)), label="%.1f" % ach)
    ax3.set_xlim(0.0, 1.0)
    ax3.set_xlabel("$R_{DA}$")
    ax3.set_ylabel("$X_{end} - X_{start}$ [m]")
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.yaxis.set_ticks_position('left')
    ax3.xaxis.set_ticks_position('bottom')
    ax3.locator_params(axis='y', nbins=6)

    pyplot.savefig(fig_dir + "figure6_replay_stats_1d.eps", transparent=True, dpi=300)
    pyplot.savefig(fig_dir + "figure6_replay_stats_1d.png", transparent=True, dpi=300)
    pyplot.close()


def paper_plot_2d(root_dir, params: dict):

    set_text()

    analysis_dir = root_dir + "/analysis_2d/"
    fig_dir = root_dir + "/figures/"
    da_list = params["da_list"]
    n_da = len(da_list)

    # ---------------------------------------- fig 11 ---------------------------------------- #
    trial_array = np.arange(1, params["n_train_trial"] + 1, dtype=np.int)
    trial_time = np.zeros((params["n_session"], n_da, params["n_train_trial"]))
    reward_rate = np.zeros((params["n_session"], n_da, params["n_train_trial"]))

    trial_time_up95 = np.zeros((n_da, params["n_train_trial"]))
    trial_time_bot95 = np.zeros((n_da, params["n_train_trial"]))
    trial_time_up95_suc = np.zeros((n_da, params["n_train_trial"]))
    trial_time_bot95_suc = np.zeros((n_da, params["n_train_trial"]))
    reward_rate_up95 = np.zeros((n_da, params["n_train_trial"]))
    reward_rate_bot95 = np.zeros((n_da, params["n_train_trial"]))

    first_reward_trial = np.zeros((n_da, params["n_session"]))
    first_reward_time = np.zeros((n_da, params["n_session"]), dtype=np.float)
    first_reward_time_p1 = np.zeros((n_da, params["n_session"]), dtype=np.float)

    # make data for SPSS
    r_da = np.zeros(params["n_session"] * n_da)
    time_to_reward = np.zeros(params["n_session"] * n_da)
    successful_rate = np.zeros(params["n_session"] * n_da)

    # load data
    for j, da in enumerate(da_list):
        for session in range(params["n_session"]):
            folder_name = root_dir + 'water_maze_task/da%.2f/session%d' % (da, session)
            trial_end_time = np.load(folder_name + '/trial_end_time.npy') / 1000  # ms to s
            trial_time[session, j, :] = trial_end_time
            reward_rate[session, j, :] = (trial_end_time < (params["train_simulation_time"] - 1) / 1000).astype(np.float)
            idx = np.where(trial_end_time != (params["train_simulation_time"]-1) * 0.001)[0][0]
            first_reward_trial[j, session] = idx
            first_reward_time[j, session] = np.copy(trial_end_time[idx])
            first_reward_time_p1[j, session] = np.copy(trial_end_time[idx+1])
            # data for SPSS (last lap)
            r_da[j * params["n_session"] + session] = da
            time_to_reward[j * params["n_session"] + session] = trial_end_time[-1]
            successful_rate[j * params["n_session"] + session] = (trial_end_time[-1] < (params["train_simulation_time"] - 1) / 1000).astype(np.float)

        for trial in range(params["n_train_trial"]):
            t_dist = stats.t(loc=trial_time[:, j, trial].mean(), scale=np.sqrt(trial_time[:, j, trial].var() /
                                                                               params["n_session"]), df=params["n_session"] - 1)
            trial_time_up95[j, trial], trial_time_bot95[j, trial] = t_dist.interval(alpha=0.95)

            data = trial_time[:, j, trial]
            t_dist = stats.t(loc=data[data != 19.999].mean(), scale=np.sqrt(data[data != 19.999].var() /
                                                                            np.sum(data != 19.999)), df=np.sum(data != 19.999) - 1)
            trial_time_up95_suc[j, trial], trial_time_bot95_suc[j, trial] = t_dist.interval(alpha=0.95)
            p_hat = np.sum(reward_rate[:, j, trial]) / params["n_session"]
            reward_rate_up95[j, trial] = p_hat + 1.96 * np.sqrt(p_hat*(1-p_hat)/params["n_session"])    # 95% confidence interval, z(0.025) = 1.96
            reward_rate_bot95[j, trial] = p_hat - 1.96 * np.sqrt(p_hat*(1-p_hat)/params["n_session"])

    # without failed trial
    time_out_idx = trial_time == 19.999
    non_time_out_cnt = np.sum(~time_out_idx, axis=0)
    trial_time_suc = np.copy(trial_time)
    trial_time_suc[time_out_idx] = 0
    trial_time_suc = np.sum(trial_time_suc, axis=0) / non_time_out_cnt

    trial_grid = np.zeros((params["n_session"], params["n_train_trial"]), dtype=np.int)
    for trial in range(1, params["n_train_trial"]):
        trial_grid[:, trial] = trial + 1

    r_da, time_to_reward = [], []
    for j, da in enumerate(da_list):
        last_lap = trial_time[:, j, -1]
        time_out_idx = last_lap == 19.999
        r_da.append([da] * np.sum(~time_out_idx))
        time_to_reward.append(last_lap[~time_out_idx])

    # calc mean
    reward_rate = np.mean(reward_rate, axis=0)

    # ---------------------------------------- figure 7 ---------------------------------------- #

    # trial time plot
    fig = pyplot.figure(figsize=(cm2inch(15.0), cm2inch(9.2)))
    fig.subplots_adjust(wspace=0.4, hspace=0.25, left=0.1, right=0.95, bottom=0.1, top=0.95)

    ax1 = fig.add_subplot(2, 3, 2)
    for j, da in enumerate(da_list):
        ax1.plot(trial_array, trial_time_suc[j], "-", color=cm.jet(j / (n_da-1)), label="%.1f" % da)
        ax1.fill_between(trial_array, trial_time_up95_suc[j], trial_time_bot95_suc[j], facecolor=cm.jet(j / (n_da-1)), alpha=0.3)
    ax1.set_ylabel('Time to reward [s]')
    ax1.set_xlabel('Trial#')
    ax1.set_ylim(5, 15)
    ax1.set_xlim(1, params["n_train_trial"])
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.locator_params(axis='y', nbins=6)
    ax1.set_xticks([1, 12, 24, 36])
    ax1.set_yticks([5, 10, 15])

    # reward rate plot
    ax2 = fig.add_subplot(2, 3, 3)
    for j, da in enumerate(da_list):
        ax2.plot(trial_array, reward_rate[j] * 100, "-", color=cm.jet(j / (n_da-1)), label="%.1f" % da)
        ax2.fill_between(trial_array, reward_rate_up95[j] * 100, reward_rate_bot95[j] * 100, facecolor=cm.jet(j / (n_da-1)), alpha=0.3)
    ax2.set_ylabel('Reward acquisition rate [%]')
    ax2.set_xlabel('Trial#')
    ax2.set_ylim(0, 100)
    ax2.set_xlim(1, params["n_train_trial"])
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.yaxis.set_ticks_position('left')
    ax2.xaxis.set_ticks_position('bottom')
    ax2.locator_params(axis='y', nbins=6)
    ax2.set_xticks([1, 12, 24, 36])

    # axis
    theta_axis = np.linspace(0, 2 * np.pi, params["n_part"])
    theta_v_dist = np.load(analysis_dir + "theta_v_dist.npy")

    def plot_polar(obj_fig: pyplot.Figure, ax_idx: tuple, angles, values, norm=False):
        angles[-1] = angles[0]
        values[-1] = values[0]
        if norm is True:
            values /= np.sum(values)
        ax_polar = obj_fig.add_subplot(*ax_idx, polar=True)
        ax_polar.plot(angles, values, '-')
        ax_polar.fill(angles, values, alpha=0.5)
        ax_polar.tick_params(labelleft=False, left=False)
        return ax_polar

    mov_lim = 0.2
    mov_0 = np.load(root_dir + "/analysis_2d/mov_sample_da0.0.npy")
    mov_1 = np.load(root_dir + "/analysis_2d/mov_sample_da0.5.npy")
    theta_0 = np.load(root_dir + "/analysis_2d/theta_sample_da0.0.npy")
    theta_1 = np.load(root_dir + "/analysis_2d/theta_sample_da0.5.npy")
    mov_0_idx = mov_0 <= mov_lim
    mov_1_idx = mov_1 <= mov_lim
    mov_0 = mov_0[mov_0_idx]
    mov_1 = mov_1[mov_1_idx]
    theta_0 = np.mod(theta_0 + np.pi * 2, np.pi * 2)
    theta_1 = np.mod(theta_1 + np.pi * 2, np.pi * 2)
    theta_0 = theta_0[mov_0_idx]
    theta_1 = theta_1[mov_1_idx]

    ax1 = pyplot.subplot(2, 3, 4)
    ax1.hist(mov_0, alpha=0.5, bins=100, label="$R_{DA}$:0.0", density=True)
    ax1.hist(mov_1, alpha=0.5, bins=100, label="$R_{DA}$:0.5", density=True)
    ax1.legend(framealpha=0.0, fontsize=6)
    ax1.set_xlabel("Distance [m]")
    ax1.set_ylabel("Normalized frequency")

    ax2 = pyplot.subplot(2, 3, 5)
    ax2.hist(theta_0, alpha=0.5, bins=100, label="$R_{DA}$:0.0", density=True)
    ax2.hist(theta_1, alpha=0.5, bins=100, label="$R_{DA}$:0.5", density=True)
    ax2.legend(framealpha=0.0, fontsize=6)
    ax2.set_xlabel("Direction [rad]")
    ax2.set_ylabel("Normalized frequency")
    ax2.set_ylim(0.1, 0.27)
    ax2.set_xticks([0, np.pi/2, np.pi, 3/2*np.pi, 2*np.pi-theta_axis[1]])
    ax2.set_xticklabels(["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3}{2}\pi$", r"$2\pi$"])

    # polar plot and 2d histogram
    theta_v_hist = theta_v_dist[0, 0, 1].transpose()[:, :-1]
    theta_v_hist = np.roll(theta_v_hist, -int(theta_axis.size / 2), axis=1)
    plot_polar(fig, (2, 3, 6), angles=np.copy(theta_axis[:-1]), values=np.sum(theta_v_hist, axis=0), norm=True)
    theta_v_hist = theta_v_dist[1, 0, 1].transpose()[:, :-1]
    theta_v_hist = np.roll(theta_v_hist, -int(theta_axis.size / 2), axis=1)
    ax = plot_polar(fig, (2, 3, 6), angles=np.copy(theta_axis[:-1]), values=np.sum(theta_v_hist, axis=0), norm=True)
    ax.set_xticklabels(['0', '', r"$\frac{\pi}{2}$", '', r"$\pi$", '', r"$\frac{3}{2}\pi$", ''])
    ax.legend(labels=["0.0", "0.5"], title="$R_{DA}$", bbox_to_anchor=(1.005, 1.2), loc="upper left", framealpha=0.0)
    pyplot.tight_layout()
    pyplot.savefig(fig_dir + "/figure7_field_shift_2d.tiff", transparent=True, dpi=300)
    pyplot.savefig(fig_dir + "/figure7_field_shift_2d.png", transparent=True, dpi=300)
    pyplot.close(fig)

    # ---------------------------------------- figure 8 ---------------------------------------- #
    ach = 2.5
    fig = pyplot.figure(figsize=(cm2inch(15.0), cm2inch(15.0)))
    fig.subplots_adjust(wspace=0.4, hspace=0.25, left=0.1, right=0.95, bottom=0.1, top=0.95)
    ax1 = fig.add_subplot(2, 2, 2)
    replay_freq = np.load(analysis_dir + "f_ach%.1f/replay_freq_2d.npy" % ach)
    y = np.nanmean(replay_freq, axis=1)
    y_err = np.nanstd(replay_freq, axis=1) / np.sqrt(params["n_session"])
    ax1.bar(da_list, y, yerr=y_err, width=0.25, capsize=3, color="grey")
    _, p = ttest_ind(replay_freq[0, :], replay_freq[1, :], equal_var=False)
    print("p value = %f" % p)
    barplot_annotate_brackets(0, 1, p, da_list, y, yerr=y_err)
    ax1.set_xlabel("$R_{DA}$")
    ax1.set_ylabel("Replay frequency [1/s]")
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('none')
    ax1.set_xticks([0.0, 0.5])
    ax1.set_ylim(0.0, 0.06)
    ax1.locator_params(axis='y', nbins=6)

    start_pos = np.load(analysis_dir + "f_ach%.1f/start_pos.npy" % ach)
    end_pos = np.load(analysis_dir + "f_ach%.1f/end_pos.npy" % ach)

    xy = np.arange(0, 1.5 + 0.1, 0.1)
    X, Y = np.meshgrid(xy, xy)
    start_map, end_map = X * 0.0, X * 0.0
    for x, y in start_pos:
        start_map[np.searchsorted(xy, x, side="right") - 1, np.searchsorted(xy, y, side="right") - 1] += 1
    for x, y in end_pos:
        end_map[np.searchsorted(xy, x, side="right") - 1, np.searchsorted(xy, y, side="right") - 1] += 1

    ax2 = fig.add_subplot(2, 2, 3)
    heatmap(np.flipud(start_map), cmap=cm.Blues, cbar=True, square=True, cbar_kws={"shrink": .8, 'label': 'Frequency'})
    ax2.plot([0, X.shape[0], X.shape[0], 0, 0], [0, 0, X.shape[0], X.shape[0], 0], "k-")
    ax2.set_title("Replay start point")
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax3 = fig.add_subplot(2, 2, 4)
    heatmap(np.flipud(end_map), cmap=cm.Reds, cbar=True, square=True, cbar_kws={"shrink": .8, 'label': 'Frequency'})
    ax3.plot([0, X.shape[0], X.shape[0], 0, 0], [0, 0, X.shape[0], X.shape[0], 0], "k-")
    ax3.set_title("Replay end point")
    ax3.set_xticks([])
    ax3.set_yticks([])

    pyplot.savefig(fig_dir + "figure8_replay_stats_2d.eps", transparent=True, dpi=300)
    pyplot.savefig(fig_dir + "figure8_replay_stats_2d.png", transparent=True, dpi=300)
    pyplot.close()


def calc_spike_skewness_x_axis(spikes: sa.SpikeData, trajectory: sa.TrajectoryData, time_window: list):
    skew_list, width_list, com_list = np.zeros(spikes.n_cells), np.zeros(spikes.n_cells), np.zeros(spikes.n_cells)
    for i in range(spikes.n_cells):
        # search cell's spike data
        index = np.where(spikes.cell_id == spikes.member[i])[0]
        # cell_spike = np.floor(spikes.spike_timing[index]).astype(np.int32)  # ms
        cell_spike = spikes.spike_timing[index]  # ms
        # 解析対象の時間窓が与えられている場合は時間外の発火は除外する
        cell_spike = cell_spike[(time_window[0] < cell_spike) & (cell_spike <= time_window[1])]
        if cell_spike.size:
            x = trajectory.x[np.searchsorted(trajectory.t, cell_spike, side="right") - 1]
            skew_list[i] = stats.skew(x)
            width_list[i] = x[-1] - x[0]
            com_list[i] = np.mean(x)
        else:
            skew_list[i] = np.nan
            width_list[i] = np.nan
            com_list[i] = np.nan
    return skew_list, width_list, com_list


def skew_map_plot_1d(root_dir: str, params: dict):

    import pickle
    from scipy import ndimage
    from ec_hpc_net import EcHpc

    n_trial = params["n_train_trial"]
    da_list = params["da_list"]
    fig_dir = root_dir + "/figures/"

    with open(root_dir + 'linear_track_task/da%.2f/session%d' % (da_list[0], 0) + "/snn_trial%d.pickle" % (n_trial - 1), "rb") as f:
        snn: EcHpc = pickle.load(f)

    # output data array
    delta_x = 0.01   # m
    bin_time = delta_x / params["rat_speed"]    # m / (m/s) = sec
    x_axis = np.arange(0.0, 1.5+delta_x, delta_x)

    # load data
    da, session = 0.5, 0
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

    for j in range(1, n_trial):
        t_s, t_as = t_sum[j], t_sum[j] + start_act
        place_tot = np.concatenate((place_tot[:np.searchsorted(place_t_vec, t_s)], place_tot[np.searchsorted(place_t_vec, t_as):]))
        spike_tot = np.concatenate((spike_tot[:np.searchsorted(spike_t_vec, t_s)], spike_tot[np.searchsorted(spike_t_vec, t_as):]))
        place_t_vec = np.concatenate((place_t_vec[:np.searchsorted(place_t_vec, t_s)], place_t_vec[np.searchsorted(place_t_vec, t_as):]))
        spike_t_vec = np.concatenate((spike_t_vec[:np.searchsorted(spike_t_vec, t_s)], spike_t_vec[np.searchsorted(spike_t_vec, t_as):]))

    p_edge = [np.searchsorted(place_t_vec, t_sum[j]) for j in range(n_trial + 1)]
    s_edge = [np.searchsorted(spike_t_vec, t_sum[j]) for j in range(n_trial + 1)]

    # find skewed cell
    tr = sa.TrajectoryReader(place_tot[p_edge[n_trial-1]:p_edge[n_trial]], env_size=[0, 0, 1.5, 0.2])
    sr = sa.SpikeReader(spike_tot[s_edge[n_trial-1]:s_edge[n_trial]])
    trajectory = tr.make_trajectory()
    pyr = sr.make_group(snn.pyramidal.gid)
    skew, width, com = calc_spike_skewness_x_axis(pyr, trajectory, time_window=[0, 1000000])
    skew_cells = np.where(skew < -0.3)[0]

    # ---------------------------------------- fig 6 ---------------------------------------- #

    fig = pyplot.figure(figsize=(cm2inch(13.0), cm2inch(13.0)))
    fig.subplots_adjust(wspace=0.3, hspace=0.25, left=0.1, right=0.95, bottom=0.1, top=0.95)
    cell_id = skew_cells[0]
    ax2 = fig.add_subplot(221)
    for i, trial in enumerate([0, n_trial - 1]):
        tr = sa.TrajectoryReader(place_tot[p_edge[trial]:p_edge[trial + 1]], env_size=[0, 0, 1.5, 0.2])
        sr = sa.SpikeReader(spike_tot[s_edge[trial]:s_edge[trial + 1]])
        trajectory = tr.make_trajectory()
        pyr = sr.make_group(snn.pyramidal.gid)
        index = np.where(pyr.cell_id == pyr.member[cell_id])[0]
        trial_skew, _, _ = calc_spike_skewness_x_axis(pyr, trajectory, time_window=[0, 1000000])
        cell_spike = pyr.spike_timing[index]
        # make 2D-space histogram of spikes
        t = np.searchsorted(trajectory.t, cell_spike, side='right') - 1
        x = trajectory.x[t]
        x_hist = np.histogram(x, bins=x_axis)[0]
        x_smoothed = ndimage.gaussian_filter1d(x_hist.astype(np.float) / bin_time, sigma=2.5)
        if i == 0:
            pyplot.plot(x_axis[:-1] * 100, x_smoothed, "k", label="1st lap")
            pyplot.text(68.5, 5, str(np.round(trial_skew[cell_id], 2)), fontsize=10, color="k")
        else:
            pyplot.plot(x_axis[:-1] * 100, x_smoothed, "r", label="10th lap")
            pyplot.text(65, 15, str(np.round(trial_skew[cell_id], 2)), fontsize=10, color="r")
            pyplot.xlim([max(0, x[0]*100-10), min(150, x[-1]*100+10)])
    pyplot.legend(loc="upper left", framealpha=0.0)
    ax2.set_ylim(0, 25)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.yaxis.set_ticks_position('left')
    ax2.xaxis.set_ticks_position('bottom')
    ax2.locator_params(axis='y', nbins=6)
    ax2.set_ylabel("Firing Rate [Hz]")
    ax2.set_xlabel("Location [cm]")

    # cell_id = skew_cells[6]
    # ax2 = fig.add_subplot(222)
    # for i, trial in enumerate([0, n_trial - 1]):
    #     tr = sa.TrajectoryReader(place_tot[p_edge[trial]:p_edge[trial + 1]], env_size=[0, 0, 1.5, 0.2])
    #     sr = sa.SpikeReader(spike_tot[s_edge[trial]:s_edge[trial + 1]])
    #     trajectory = tr.make_trajectory()
    #     pyr = sr.make_group(snn.pyramidal.gid)
    #     trial_skew, _, _ = calc_spike_skewness_x_axis(pyr, trajectory, time_window=[0, 1000000])
    #     index = np.where(pyr.cell_id == pyr.member[cell_id])[0]
    #     cell_spike = pyr.spike_timing[index]
    #     # make 2D-space histogram of spikes
    #     t = np.searchsorted(trajectory.t, cell_spike, side='right') - 1
    #     x = trajectory.x[t]
    #     x_hist = np.histogram(x, bins=x_axis)[0]
    #     x_smoothed = ndimage.gaussian_filter1d(x_hist.astype(np.float) / bin_time, sigma=2.5)
    #     if i == 0:
    #         pyplot.plot(x_axis[:-1] * 100, x_smoothed, "r", label="1st lap")
    #         pyplot.text(100, 6, str(np.round(trial_skew[cell_id], 2)), fontsize=12, color="k")
    #         # pyplot.text(71, 5, str(np.round(, 2)))
    #     else:
    #         pyplot.plot(x_axis[:-1] * 100, x_smoothed, "b", label="10th lap")
    #         pyplot.text(90, 7.8, str(np.round(trial_skew[cell_id], 2)), fontsize=12, color="r")
    #         pyplot.xlim([max(0, x[0]*100-10), min(150, x[-1]*100+10)])
    # pyplot.legend(loc="upper left", framealpha=0.0)
    # ax2.spines['right'].set_visible(False)
    # ax2.spines['top'].set_visible(False)
    # ax2.yaxis.set_ticks_position('left')
    # ax2.xaxis.set_ticks_position('bottom')
    # ax2.locator_params(axis='y', nbins=6)
    # ax2.set_ylabel("Firing Rate [Hz]")
    # ax2.set_xlabel("Location [cm]")

    pyplot.savefig(fig_dir + "figure5_place_field_stats_b.eps", transparent=True, dpi=300)
    pyplot.savefig(fig_dir + "figure5_place_field_stats_b.png", transparent=True, dpi=300)
    pyplot.close()


def figure_7a(train_dir: str, root_dir: str, params: dict):

    place_tot: np.ndarray = np.load(train_dir + "/place_total.npy")
    place_tot_list = []

    last_t = .0
    for t, x, y in place_tot:
        if t - last_t > 4.0 and x == 0.75 and y == 0.75:
            place_tot_list.append(place_tot[int(last_t * 1000)+100:int(t * 1000)-100])
            last_t = t
    else:
        place_tot_list.append(place_tot[int(last_t * 1000):])

    # plot trajectory
    fig = pyplot.figure(figsize=(cm2inch(10.), cm2inch(10.0)))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([0.0, 1.5, 1.5, 0.0, 0.0], [0.0, 0.0, 1.5, 1.5, 0.0], "-k")
    for trial, place in enumerate(place_tot_list):
        ax.plot(place[:, 1], place[:, 2], color=pyplot.cm.jet(trial / params["n_train_trial"]), alpha=0.3, lw=2)
    ax.set_xlim(0, 1.5)
    ax.set_ylim(0, 1.5)
    pyplot.axis('equal')
    fig.savefig(root_dir + 'figures/figure7a_trajectory.eps', transparent=True, dpi=300)
    fig.savefig(root_dir + 'figures/figure7a_trajectory.png', transparent=True, dpi=300)
    pyplot.close(fig)


def barplot_annotate_brackets(num1, num2, data, center,
                              height, yerr=None, dh=.05,
                              barh=.05, fs=None, maxasterix=None):

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p and text != "***":
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr is not None:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = pyplot.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y + barh, y + barh, y]
    mid = ((lx + rx) / 2, y + barh)

    pyplot.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    pyplot.text(*mid, text, **kwargs)
