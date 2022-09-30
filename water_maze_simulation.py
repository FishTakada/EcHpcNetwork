import os
import sys
import pickle
import subprocess
from tasks import replay_by_noise, water_maze_task, place_field_evaluation, trajectory_modulation
from analysis import place_cell_analysis_2d, replay_stats_2d
from replay_analysis import TrajectoryEvent
from plot_figs import paper_plot_2d, figure_7a


def main(root_dir: str):

    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(root_dir + "/analysis_2d/", exist_ok=True)
    os.makedirs(root_dir + "/figures/", exist_ok=True)

    # parameters
    params = {
              "task": "water_maze",
              "n_session": 150,
              "n_train_trial": 36,
              "n_replay_trial": 10,
              "n_trajectory_session": 100,
              "n_trajectory_trial": 10,
              "da_list": [0.0, 0.5, 1.0],
              "block_size": 256,
              "dt": 0.2,  # time step for numerical solution[ms]
              "max_spike_in_step": 2048,
              "wij_update_interval": 20,    # ms

              # parameter in simulation
              "train_simulation_time": 20 * 1000,   # ms
              "replay_simulation_time": 20 * 1000,  # ms
              "act_start": 2000,                    # ms
              "rat_speed": 0.2,     # m/s
              "f_ach": 2.5,
              "noise_rate": 0.1,    # Hz
              "env_size": [0.0, 0.0, 1.5, 1.5],

              # network parameter
              "g1_gain": 0.7,
              "g2_gain": 1.0,

              # analysis parameter(2d)
              "session_div": 2,
              "n_part": 40,
              "mov_lim": 0.2,

              # replay analysis
              "replay_analysis_start": 5,  # sec
              "pc_lim": 0.3,    # Hz
              "ripple_lim": 2.5,    # sd
              "least_path": 0.5,    # m
              "least_step": 5,
              "jump_lim": 0.5,
              }

    # save simulation parameter
    with open(root_dir + "/params.pickle", "wb") as f:
        pickle.dump(params, f)

    # ----------- MWM simulation ----------- #
    for da in params["da_list"]:
        for session in range(params["n_session"]):
            water_maze_task(root_dir, session, da, params)
            for f_ach in [2.5]:
                replay_by_noise(root_dir + "water_maze_task/", session, da, f_ach, params)

    for da in params["da_list"]:
        for session in range(params["n_session"]):
            place_field_evaluation(root_dir, da=da, session=session, net_num=0)
            place_field_evaluation(root_dir, da=da, session=session, net_num=params["n_train_trial"]-1)

    for da in [0.0, 0.5]:
        for session in range(params["n_session"]):
            trajectory_modulation(root_dir, session, da, params)

    # ----------- analysis ----------- #
    n_process = 20

    def analysis_by_mp(start, end):
        p_list = []
        for i in range(start, end):
            cmd = ["python3", "replay_analysis.py"]
            args = [root_dir, str(i)]
            p_list.append(subprocess.Popen(cmd + args))
        for p in p_list:
            p.wait()
    quo, rem = params["n_session"] // n_process, params["n_session"] % n_process
    for sect in range(quo):
        analysis_by_mp(sect * n_process, sect * n_process + n_process)
    else:
        analysis_by_mp(params["n_session"] - rem, params["n_session"])

    place_cell_analysis_2d(root_dir, params)
    replay_stats_2d(root_dir, params)

    # # ----------- plotting ----------- #
    paper_plot_2d(root_dir, params)     # figure 7 and 8
    figure_7a(root_dir + "water_maze_task/da0.50/session15", root_dir, params)    # figure 7a


if __name__ == "__main__":
    main(sys.argv[1])
    # main("/output_dir/")
