import sys
from tasks import replay_by_noise, linear_track_task, free_exploration
from analysis import place_weight_analysis, place_cell_analysis_1d, replay_stats_1d
from replay_analysis import TrajectoryEvent
import subprocess
from plot_figs import paper_plot_1d, skew_map_plot_1d, figure1
import pickle
import os


def main(root_dir: str):

    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(root_dir + "/analysis/", exist_ok=True)
    os.makedirs(root_dir + "/figures/", exist_ok=True)

    # parameters
    params = {
              "task": "linear_track",
              "n_session": 100,
              "n_train_trial": 10,
              "n_replay_trial": 10,
              "da_list": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
              "block_size": 256,    # block size for CUDA
              "dt": 0.2,    # time step for numerical solution[ms]
              "max_spike_in_step": 2048,
              "wij_update_interval": 20,    # ms

              # parameter in simulation
              "train_simulation_time": 20 * 1000,   # ms
              "replay_simulation_time": 10 * 1000,  # ms
              "act_start": 2000,                    # ms
              "rat_speed": 0.2,     # m/s
              "f_ach": 2.5,
              "noise_rate": 0.1,    # Hz
              "env_size": [0.0, 0.0, 1.5, 0.2],

              # network parameter
              "g1_gain": 0.7,
              "g2_gain": 1.0,

              # analysis parameter
              "space_bin_size": 0.025,  # m
              "weight_width": 0.6,   # m
              "da_fix": 0.5,
              "trial_fix": 5,
              "x_bin": 0.1,   # m

              # replay analysis
              "replay_analysis_start": 5,  # sec
              "pc_lim": 0.3,    # Hz
              "ripple_lim": 2.5,    # sd
              "least_path": 0.5,    # m
              "least_step": 5,
              "jump_lim": 0.5,  # m

              }

    # save parameter
    with open(root_dir + "/params.pickle", "wb") as f:
        pickle.dump(params, f)

    # ----------- linear track simulation ----------- #
    for da in params["da_list"]:
        for session in range(params["n_session"]):
            linear_track_task(root_dir, session, da, params)    # track running simulation
            for f_ach in [1.0, 1.5, 2.0, 2.5, 3.0]:
                replay_by_noise(root_dir + "linear_track_task/", session, da, f_ach, params)    # resting state simulation

    # # ----------- analysis ----------- #
    place_weight_analysis(root_dir, params)
    place_cell_analysis_1d(root_dir, params)

    # run subprocess for replay analysis
    n_process = 20  # number of process

    def analysis_by_mp(start, end):
        p_list = []
        for i in range(start, end):
            cmd = ["python3", "replay_analysis.py"]
            args = [root_dir, str(i)]
            p_list.append(subprocess.Popen(cmd + args))
        for p in p_list:
            p.wait()
    quo, rem = params["n_session"] // n_process, params["n_session"] % n_process
    for f_ach in [1.0, 1.5, 2.0, 2.5, 3.0]:
        params["f_ach"] = f_ach
        with open(root_dir + "/params.pickle", "wb") as f:
            pickle.dump(params, f)
        for sect in range(quo):
            analysis_by_mp(sect * n_process, sect * n_process + n_process)
        else:
            analysis_by_mp(params["n_session"] - rem, params["n_session"])
        replay_stats_1d(root_dir, params)

    # plotting
    paper_plot_1d(root_dir, params)     # figs 3-6
    skew_map_plot_1d(root_dir, params)  # figure 5b

    free_exploration(root_dir + "/free_exploration/")
    figure1(root_dir + "/free_exploration/")


if __name__ == "__main__":
    main(sys.argv[1])
    # main("/output_dir/")
