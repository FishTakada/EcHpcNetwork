import os
import pickle

import matplotlib as mpl
import numpy as np

mpl.use('Agg')
from matplotlib import pyplot
from progressbar import ProgressBar

# noinspection PyUnresolvedReferences
import cusnn as cs
from ec_hpc_net import EcHpcSimple

import cusnn.analysis as sa
from agent import WaterMazeAgent, LinearTrackAgent, FreeExplorationAgent
from env import Box

pi2 = np.pi * 2


def linear_track_task(root_dir: str, session: int, da: float, params: dict):
    n_trial = params["n_train_trial"]

    # make directory
    folder_name = root_dir + "linear_track_task/da%.2f/session%d" % (da, session)
    os.makedirs(folder_name, exist_ok=True)

    # make environment
    env = Box(x_max=1.5, y_max=0.2)
    env.reward_range = 0.2
    env.goal[0] = 1.5
    env.goal[1] = 0.1

    # make network
    snn = EcHpcSimple(params={"conn_seed": session})

    # session total log
    step_tot = 0
    place_tot = []
    spike_tot = []
    trial_end_time = np.ones(n_trial)
    head, tail = -params["weight_width"] - 0.5 * params["space_bin_size"], params["weight_width"] - 0.5 * params["space_bin_size"]
    place_bins = np.arange(head, tail + params["space_bin_size"], params["space_bin_size"])
    place_weight = np.zeros((n_trial, place_bins.shape[0] - 1))

    # trial loop
    for trial in range(n_trial):
        print("\nda %.2f, session %d, trial %d" % (da, session, trial))

        # make network
        simulator = cs.Simulator(snn, dt=params["dt"], max_spike_in_step=params["max_spike_in_step"], block_size=params["block_size"],
                                 wij_update_interval=params["wij_update_interval"], silent=True, method="rk4")
        simulator.snn.has_update_weight = False  # initially STDP is disabled
        place = []
        rewarded_time = 10 ** 9

        # make agent
        rat = LinearTrackAgent(snn_n_dir=snn.n_dir, n_dir=snn.n_action)

        # initialize environment and agent
        rat.sp_prob = 0.0
        rat.set_velocity(0.0)
        rat.set_angle(0.0)
        observation = env.reset(ini_x=0.2, ini_y=0.1)  # state (x[m], y[m])

        # add electrode for the initial input to the grid cells
        simulator.add_electrode(snn.grid_1, t_start=100, t_end=200, array=rat.make_initial_input_to_grid(snn.grid_1, snn.ny))
        simulator.add_electrode(snn.grid_2, t_start=100, t_end=200, array=rat.make_initial_input_to_grid(snn.grid_2, snn.ny))

        # add electrode for the initial input to the action neurons
        init_dir = np.searchsorted(snn.action_dir, rat.angle, side="right") - 1
        ac_init = np.zeros(snn.action_neuron.n_cells)
        ac_init[init_dir] = 0.7
        ac_init[(init_dir + 1) % snn.n_action] = 0.7
        ac_init[init_dir - 1] = 0.7
        simulator.add_electrode(snn.action_neuron, t_start=100, t_end=200, array=ac_init)

        # simulation main loop
        progress = ProgressBar(min=0, max_value=params["train_simulation_time"])
        for step in range(params["train_simulation_time"]):

            # update progressbar
            if (step + 1) % 1000 == 0:
                progress.update(step + 1)

            # enable stdp
            if step == params["act_start"]:
                rat.set_velocity(params["rat_speed"])
                simulator.snn.has_update_weight = True
                simulator.write_to_var("trace", val=0.0)

            # the rat model starts exploration
            if step >= params["act_start"]:
                # get action
                action = rat.get_action(env)
                observation, reward, done = env.step(action, show=False)
                # update network
                if done and (rewarded_time > 10 ** 8):
                    simulator.add_to_var("da", da, cell_group=snn.pyramidal)
                    rewarded_time = step

            # update network
            simulator.add_step_current(snn.head_1, array=rat.make_input_to_hd(gain=params["g1_gain"]))
            simulator.add_step_current(snn.head_2, array=rat.make_input_to_hd(gain=params["g2_gain"]))
            simulator.step_advance(step)

            # write log data
            place.append(np.array([step * 0.001, observation.x, observation.y], dtype=np.float32))  # Time[s] X[m] Y[m]

            # end condition of the trial
            if step >= rewarded_time + 200 or step == params["train_simulation_time"] - 1:
                # stack spike data
                spike = simulator.get_spike_data()
                spike[:, 0] += step_tot * 0.001
                spike_tot.append(spike)
                # stack place data
                place = np.vstack(place)
                place[:, 0] += step_tot * 0.001
                place_tot.append(place)
                # write trial time
                trial_end_time[trial] = step
                step_tot += step + 1
                break

        # -------------------- save trial results -------------------- #
        simulator.copy_wij_to_snn(snn)  # update snn wij
        if trial == n_trial - 1:
            with open(folder_name + "/snn_trial%d.pickle" % trial, "wb") as f:
                pickle.dump(snn, f)

        # -------------------- analysis of trial results -------------------- #
        SR = sa.SpikeReader(simulator.get_spike_data())
        trial_place = np.copy(place)
        trial_place[:, 0] -= trial_place[0, 0]
        TR = sa.TrajectoryReader(trial_place, env_size=[0, 0, env.x_max, env.y_max])
        traj = TR.make_trajectory()

        # analysis of connection between pyramidal and pyramidal
        pyr_center = sa.calc_center(
            sa.calc_ratemap2d(SR.make_group(snn.pyramidal.gid), traj, time_window=[params["act_start"] * 0.001, trial_end_time[trial] * 0.001]))
        dist_pyr2pyr = np.zeros((snn.pyramidal.n_cells, snn.pyramidal.n_cells))  # [x, y, peak_hz]
        for i in range(snn.pyramidal.n_cells):
            dist_pyr2pyr[i, :] = pyr_center[:, 0] - pyr_center[i, 0]  # dist = post_x - pre_x

        # get weight of pyramidal to action
        wij_ind = snn.wij.index[snn.pyramidal.cell_type, snn.pyramidal.cell_type]
        wij_pyr2pyr = np.copy(np.reshape(snn.wij.w[wij_ind:(wij_ind + snn.pyramidal.n_cells ** 2)], (snn.pyramidal.n_cells, snn.pyramidal.n_cells)))
        # remove not connected array
        dist_pyr2pyr = dist_pyr2pyr[np.nonzero(wij_pyr2pyr)]
        wij_pyr2pyr = wij_pyr2pyr[np.nonzero(wij_pyr2pyr)]

        # remove not fired neuron
        wij_pyr2pyr = wij_pyr2pyr[~np.isnan(dist_pyr2pyr)]
        dist_pyr2pyr = dist_pyr2pyr[~np.isnan(dist_pyr2pyr)]

        # calc place weight func
        place_index = np.digitize(dist_pyr2pyr, bins=place_bins)
        for x in range(1, len(place_bins)):
            ids = place_index == x
            if ids.any():
                place_weight[trial][x - 1] = np.mean(wij_pyr2pyr[ids])

        # close objects
        del simulator

    # stack data of all sessions
    place_tot = np.vstack(place_tot)
    spike_tot = np.vstack(spike_tot)

    # save data of training trials
    np.save(folder_name + "/spike_total.npy", spike_tot)
    np.save(folder_name + "/place_total.npy", place_tot)
    np.save(folder_name + "/place_weight.npy", place_weight)
    np.save(folder_name + "/trial_end_time.npy", trial_end_time)


def water_maze_task(root_dir: str, session: int, da: float, params: dict, mode="Train"):
    n_trial = params["n_train_trial"]
    simulation_time = params["train_simulation_time"]

    # make environment
    env = Box()
    env.reward_range = 0.2
    env.goal[0] = 1.3
    env.goal[1] = 1.3

    # make directory
    folder_name = root_dir + "water_maze_task/da%.2f/session%d" % (da, session)
    print(folder_name)
    os.makedirs(folder_name, exist_ok=True)

    # make network
    snn = EcHpcSimple(params={"conn_seed": session})
    goal_cnt = 0
    trial_end_time = np.ones(n_trial) * simulation_time

    # data array
    step_tot = 0
    place_tot = []
    spike_tot = []
    pyr2ac_idx = 0
    pyr2ac_sum = np.zeros((n_trial, snn.n_action))
    pyr2ac_list = []

    # trial loop
    for trial in range(n_trial):

        print("\nda %.2f, session %d, trial %d" % (da, session, trial))

        # make network
        simulator = cs.Simulator(snn, dt=params["dt"], max_spike_in_step=params["max_spike_in_step"], block_size=params["block_size"],
                                 wij_update_interval=params["wij_update_interval"], silent=True, method="rk4")
        simulator.snn.has_update_weight = False  # initially STDP is disabled
        simulator.write_to_var(var="f_ach", val=1.0)
        place = []
        rewarded_time = 10 ** 9

        # make agent
        rat = WaterMazeAgent(snn_n_dir=snn.n_dir, n_action=snn.n_action, seed=session * 1000 + trial)

        # initialize environment and agent
        rat.sp_prob = 0.0
        rat.set_velocity(0.0)
        observation = env.reset(ini_x=0.75, ini_y=0.75)  # state (x[m], y[m])

        # # initialize membrane potential(uniform, 0.0~0.05mV)
        rs = np.random.RandomState(seed=session * 1000 + trial)
        simulator.write_to_var("v", val=rs.random_sample(snn.n_cells) * 0.05)

        # add electrode for the initial input to the grid cells
        simulator.add_electrode(snn.grid_1, t_start=100, t_end=200, array=rat.make_initial_input_to_grid(snn.grid_1, snn.ny))
        simulator.add_electrode(snn.grid_2, t_start=100, t_end=200, array=rat.make_initial_input_to_grid(snn.grid_2, snn.ny))
        # add electrode for the initial input to the action neurons
        init_dir = np.searchsorted(snn.action_dir, rat.angle, side="right") - 1
        ac_init = np.zeros(snn.action_neuron.n_cells)
        ac_init[init_dir] = 0.7
        ac_init[(init_dir + 1) % snn.n_action] = 0.7
        ac_init[init_dir - 1] = 0.7
        simulator.add_electrode(snn.action_neuron, t_start=100, t_end=200, array=ac_init)

        # simulation main loop
        progress = ProgressBar(min=0, max_value=simulation_time)
        for step in range(simulation_time):

            # update progressbar
            if step % 1000 == 0:
                progress.update(step)

            # start act
            if step == params["act_start"]:
                rat.set_velocity(params["rat_speed"])
                if mode == "Train":
                    simulator.snn.has_update_weight = True
                simulator.write_to_var("trace", val=0.0)

            # the rat model starts exploration
            if step >= params["act_start"]:
                # get action
                action_spk_now = np.isin(snn.action_neuron.gid, simulator.get_spike_now()[:, 1])
                action = rat.get_action(env, action_spk_now)  # action (velocity[m/s], omega[rad/s])
                observation, reward, done = env.step(action, show=False)
                # update network
                if done and (rewarded_time > 10 ** 8):
                    simulator.add_to_var("da", 0.005, cell_group=snn.action_neuron)
                    simulator.add_to_var("da", da, cell_group=snn.pyramidal)
                    rewarded_time = step
                    goal_cnt += 1

            # update network
            simulator.add_step_current(snn.action_neuron, array=rat.make_input_to_ac(observation.x, observation.y))
            simulator.add_step_current(snn.head_1, array=rat.make_input_to_hd(gain=params["g1_gain"]))
            simulator.add_step_current(snn.head_2, array=rat.make_input_to_hd(gain=params["g2_gain"]))
            simulator.step_advance(step)

            # write current place
            place.append(np.array([step * 0.001, observation.x, observation.y], dtype=np.float32))  # Time[s] X[m] Y[m]

            # end condition of the trial
            if step >= rewarded_time + rat.action_choice_interval or step == simulation_time - 1:
                # stack spike data
                spike = simulator.get_spike_data()
                spike[:, 0] += step_tot * 0.001
                spike_tot.append(spike)
                # stack place data
                place = np.vstack(place)
                place[:, 0] += step_tot * 0.001
                place_tot.append(place)
                # write trial time
                trial_end_time[trial] = step
                step_tot += step + 1
                break

        # close objects
        simulator.copy_wij_to_snn(snn)  # update snn wij
        del simulator

        # save pyramidal 2 action neuron weight
        ind = snn.wij.index[snn.pyramidal.cell_type, snn.action_neuron.cell_type]
        pyr2ac = snn.wij.w[ind:(ind + snn.pyramidal.n_cells * snn.action_neuron.n_cells)] \
            .reshape(snn.pyramidal.n_cells, snn.action_neuron.n_cells)
        pyr2ac_list.append(np.copy(pyr2ac))
        pyr2ac_idx += 1

        # save network model
        if trial == (n_trial - 1) and mode == "Train":
            with open(folder_name + '/snn_trial%d.pickle' % trial, 'wb') as f:
                pickle.dump(snn, f)

    # plot trajectory
    fig, ax = pyplot.subplots()
    ax.plot([0.0, 1.5, 1.5, 0.0, 0.0], [0.0, 0.0, 1.5, 1.5, 0.0], "-k")
    for trial, place in enumerate(place_tot):
        ax.plot(place[:, 1], place[:, 2], color=pyplot.cm.jet(trial / n_trial), alpha=0.3)
    ax.set_xlim([0, 1.5])
    ax.set_ylim([0, 1.5])
    pyplot.axis('equal')
    fig.savefig(root_dir + '/figures/trajectory_da%.2f_session%d.png' % (da, session))
    pyplot.close(fig)

    # stack data of all sessions
    place_tot = np.vstack(place_tot)
    spike_tot = np.vstack(spike_tot)
    for trial in range(n_trial):
        pyr2ac_sum[trial] = np.sum(pyr2ac_list[trial], axis=0)

    # save total log
    np.save(folder_name + "/pyr2ac_w.npy", pyr2ac_sum)
    np.save(folder_name + '/spike_total.npy', spike_tot)
    np.save(folder_name + '/place_total.npy', place_tot)
    np.save(folder_name + "/trial_end_time.npy", trial_end_time)


def replay_by_noise(root_dir: str, session: int, da: float, f_ach: float, params: dict):
    # make directory
    folder_name = root_dir + 'da%.2f/session%d' % (da, session)
    os.makedirs(folder_name, exist_ok=True)

    # make network
    with open(folder_name + '/snn_trial%d.pickle' % (params["n_train_trial"] - 1), 'rb') as f:
        snn: EcHpcSimple = pickle.load(f)

    # Ach modulation
    # pyr->pyr
    ind = snn.wij.index[snn.pyramidal.cell_type, snn.pyramidal.cell_type]
    snn.wij.w[ind:(ind + snn.pyramidal.n_cells ** 2)] *= f_ach

    for stim_seed in range(params["n_replay_trial"]):

        # make simulator
        simulator = cs.Simulator(snn, dt=params["dt"], max_spike_in_step=params["max_spike_in_step"], block_size=params["block_size"],
                                 wij_update_interval=params["wij_update_interval"], silent=True, method="rk4")
        simulator.enable_stdp = False  # STDP is disabled
        simulator.write_to_var(var="f_ach", val=f_ach)

        # set seed
        rs = np.random.RandomState(stim_seed)

        # run simulation
        print("\nSimulation Time = " + str(params["replay_simulation_time"]) + 'ms')
        p = ProgressBar(min=0, max_value=params["replay_simulation_time"])
        for step in range(0, params["replay_simulation_time"]):
            # update progressbar
            if step % 1000 == 0:
                p.update(step)
            if 1000 <= step:
                i_injection = np.zeros(snn.pyramidal.n_cells, dtype=np.float32)
                noise = rs.random_sample(snn.pyramidal.n_cells)
                i_injection[noise < (params["noise_rate"] * 0.001)] = 10.0
                simulator.add_step_current(snn.pyramidal, array=i_injection)

            simulator.step_advance(n_step=step)

        # close files
        simulator.save_spike_data(folder_name + '/replay_spike%d_fach%.1f.npy' % (stim_seed, f_ach))
        del simulator


def free_exploration(folder_name: str):
    # make directory
    os.makedirs(folder_name, exist_ok=True)
    traj_seed = 5

    # parameters
    syn_delay = 1
    block_size = 128
    T = 400 * 1000
    start_act = 2000
    g1_gain = 0.7
    g2_gain = 1.0

    # make objects
    snn = EcHpcSimple({})
    env = Box(x_max=1.5, y_max=1.5)
    env.dt = syn_delay / 1000
    rat = FreeExplorationAgent(snn_n_dir=snn.n_dir, n_dir=36, seed=traj_seed)
    simulator = cs.Simulator(snn, block_size=block_size, syn_delay=syn_delay, save_cu_code=True, method="rk4")
    simulator.snn.has_update_weight = False  # STDP is disabled
    place_log = np.zeros((T, 3), dtype=np.float32)

    # initialize rat
    rat.sp_prob = 0.0  # [0~1]
    rat.set_velocity(0.0)  # [m/s]
    rat.set_angle(0)

    # initialize environment
    observation = env.reset(ini_x=0.75, ini_y=0.75)  # state (x[m], y[m])

    # add electrode for the initial input of the grid cells
    simulator.add_electrode(snn.grid_1, t_start=100, t_end=200, array=rat.make_initial_input_to_grid(snn.grid_1, snn.ny))
    simulator.add_electrode(snn.grid_2, t_start=100, t_end=200, array=rat.make_initial_input_to_grid(snn.grid_2, snn.ny))

    # simulation main loop
    simulation_p = ProgressBar(min=0, max_value=int(T / syn_delay))
    for step in range(int(T / syn_delay)):

        # update progressbar
        if step % 1000 == 0:
            simulation_p.update(step)
        if step == start_act:
            rat.set_velocity(0.2)  # [m/s]

        # the rat model starts exploration
        if step > start_act:
            action = rat.get_action(env)  # action (velocity[m/s], omega[rad/s])
            observation, reward, done = env.step(action, show=False)
            # reset point
            if 0.72 < observation.x < 0.78 and 0.72 < observation.y < 0.78:
                simulator.add_step_current(snn.grid_1, array=rat.make_initial_input_to_grid(snn.grid_1, snn.ny) * 0.5)
                simulator.add_step_current(snn.grid_2, array=rat.make_initial_input_to_grid(snn.grid_2, snn.ny) * 0.5)

        # update network state
        simulator.add_step_current(snn.head_1, array=rat.make_input_to_hd(gain=g1_gain))
        simulator.add_step_current(snn.head_2, array=rat.make_input_to_hd(gain=g2_gain))
        simulator.step_advance(step)
        place_log[step] = np.array([step * syn_delay * 0.001, observation.x, observation.y], dtype=np.float32)  # Time[ms] X[m] Y[m]

    # ----------------------- spike data analysis ----------------------- #

    # read data
    SR = sa.SpikeReader(simulator.get_spike_data())
    TR = sa.TrajectoryReader(place_log, env_size=[0, 0, env.x_max, env.y_max])
    traj = TR.make_trajectory()

    # plot trajectory
    pyplot.plot(traj.x, traj.y)
    pyplot.plot([0.0, 1.5, 1.5, 0.0, 0.0], [0.0, 0.0, 1.5, 1.5, 0.0], "-k")
    pyplot.xlim([0, env.x_max])
    pyplot.ylim([0, env.y_max])
    pyplot.axis('equal')
    pyplot.savefig(folder_name + '/trajectory%d.png' % traj_seed)
    pyplot.close()

    # plot ratemap
    n_col, n_row = 6, 6
    cr_size = n_col * n_row
    target_name = ["grid_1", "grid_2", "pyramidal"]
    full_output = []
    save_ratemap = ["grid_1", "grid_2", "pyramidal"]
    for name in target_name:

        tar: cs.CellGroup = snn.__dict__[name]
        cg = SR.make_group(tar.gid)
        # ratemap = sa.calc_ratemap2d(cg, traj, time_window=[start_act * 0.001, T * 0.001], sigma=3)
        ratemap = sa.calc_ratemap2d_div(cg, traj, time_window=[start_act * 0.001, T * 0.001], sigma=3)

        for i in range(np.ceil(tar.n_cells / cr_size).astype(int)) if name in full_output else [0]:
            fig = pyplot.figure(figsize=(n_row * 2, n_col * 2), facecolor='w', edgecolor='w')
            fig.subplots_adjust(hspace=.01, wspace=.01)
            for j in range(cr_size):
                if cr_size * i + j >= tar.n_cells:
                    break
                sa.plot_map2d(obj_fig=fig, ax_idx=(n_col, n_row, j + 1), ratemap=ratemap, ith=cr_size * i + j)
            fig.savefig(folder_name + '/%s_%d.png' % (name, i))
            pyplot.close(fig)

        if name == "pyramidal":
            n_field = sa.calc_place_field_number(ratemap, thresh_rate=0.1)
            print("Number field (=0): %d" % np.sum(n_field == 0))
            print("Number field (=1): %d" % np.sum(n_field == 1))
            print("Number field (=2): %d" % np.sum(n_field == 2))
            print("Number field (=3): %d" % np.sum(n_field == 3))
            pc_index = sa.find_place_cell_by_field_number(ratemap, thresh_rate=0.1, n_field_thresh=1)
            peak_rate = []
            mean_rate = []
            area = []
            for index in pc_index:
                target_map = ratemap.rate[index]
                peak_rate.append(np.max(target_map))
                mean_rate.append(np.mean(target_map[target_map > peak_rate[-1] * 0.1]))
                area.append(np.sum(target_map > peak_rate[-1] * 0.1) * ratemap.bin_size ** 2)  # m^2
            print("Mean peak firing rate: %.2fHz" % np.mean(peak_rate))
            print("Mean in-field firing rate: %.2fHz" % np.mean(mean_rate))
            print("Mean area: %.2fm^2" % np.mean(area))

        # save ratemap
        if name in save_ratemap:
            with open(folder_name + "/%s_ratemap.pickle" % name, "wb") as f:
                pickle.dump(ratemap, f)

    # close objects
    env.close()


def place_field_evaluation(root_dir, session, da, net_num=0):

    goal_x, goal_y = 1.3, 1.3
    ini_pos = [[0.75, 0.75], [0.75, 0.90], [0.75, 1.05], [0.75, 1.2], [0.75, 1.3], [0.9, 0.75], [1.05, 0.75], [1.2, 0.75], [1.3, 0.75]]

    # make directory
    folder_name = root_dir + "/water_maze_task/da%.2f/session%d" % (da, session)

    # parameters
    syn_delay = 1
    block_size = 256
    start_act = 2000
    g1_gain = 0.7
    g2_gain = 1.0

    # make objects
    if net_num == 0:
        snn = EcHpcSimple(params={"conn_seed": session})
    else:
        with open(folder_name + "/snn_trial35.pickle", "rb") as f:
            snn: EcHpcSimple = pickle.load(f)

    place_tot = []
    spike_tot = []
    step_tot = 0

    for init_x, init_y in ini_pos:

        L = np.sqrt((goal_x - init_x) ** 2 + (goal_y - init_y) ** 2)
        head_direction = np.arctan2(goal_y - init_y, goal_x - init_x)  # -pi~pi[rad]
        head_direction = np.mod(head_direction + np.pi * 2, np.pi * 2)  # 0~2pi[rad]
        T = int(L / 0.2 * 1000) + start_act

        place = []
        env = Box(x_max=1.5, y_max=1.5)
        env.dt = syn_delay / 1000
        rat = FreeExplorationAgent(snn_n_dir=snn.n_dir, n_dir=36, seed=0)
        simulator = cs.Simulator(snn, block_size=block_size, syn_delay=syn_delay, save_cu_code=True, method="rk4")
        simulator.snn.has_update_weight = False  # STDP is disabled

        # initialize rat
        rat.sp_prob = 0.0  # [0~1]
        rat.set_velocity(0.0)  # [m/s]
        rat.set_angle(head_direction)

        # initialize environment
        observation = env.reset(ini_x=0.75, ini_y=0.75)  # state (x[m], y[m])

        # add electrode for the initial input of the grid cells
        simulator.add_electrode(snn.grid_1, t_start=100, t_end=200, array=rat.make_initial_input_to_grid(snn.grid_1, snn.ny))
        simulator.add_electrode(snn.grid_2, t_start=100, t_end=200, array=rat.make_initial_input_to_grid(snn.grid_2, snn.ny))

        # simulation main loop
        simulation_p = ProgressBar(min=0, max_value=int(T / syn_delay))
        for step in range(int(T / syn_delay)):

            # update progressbar
            if step % 1000 == 0:
                simulation_p.update(step)
            if step == start_act:
                rat.set_velocity(0.2)  # [m/s]

            # the rat model starts exploration
            if step > start_act:
                action = rat.get_action(env)  # action (velocity[m/s], angle[rad])
                rat.set_angle(head_direction)  # force running
                action[1] = head_direction
                observation, reward, done = env.step(action, show=False)

            # update network state
            simulator.add_step_current(snn.head_1, array=rat.make_input_to_hd(gain=g1_gain))
            simulator.add_step_current(snn.head_2, array=rat.make_input_to_hd(gain=g2_gain))
            simulator.add_step_current(snn.action_neuron, array=np.ones(snn.action_neuron.n_cells, dtype=np.float32) * -1.5)
            simulator.step_advance(step)

            # write current place
            place.append(np.array([step * 0.001, observation.x, observation.y], dtype=np.float32))  # Time[s] X[m] Y[m]

            if step == int(T / syn_delay) - 1:
                # stack spike data
                spike = simulator.get_spike_data()
                idx = np.searchsorted(spike[:, 0], start_act * 0.001, side="right")  # remove init act
                spike = spike[idx:, :]
                spike[:, 0] += step_tot * 0.001
                spike_tot.append(spike)
                # stack place data
                place = np.vstack(place)
                idx = np.searchsorted(place[:, 0], start_act * 0.001, side="right")  # remove init act
                place = place[idx:, :]
                place[:, 0] += step_tot * 0.001
                place_tot.append(place)
                step_tot += step + 1

        del env, simulator

    # stack data of all runs
    place_tot = np.vstack(place_tot)
    spike_tot = np.vstack(spike_tot)

    # ----------------------- spike data analysis ----------------------- #
    if net_num == 0:
        np.save(folder_name + "/place_trial0.npy", place_tot)
        np.save(folder_name + "/spike_trial0.npy", spike_tot)
    else:
        np.save(folder_name + "/place_trial35.npy", place_tot)
        np.save(folder_name + "/spike_trial35.npy", spike_tot)


def trajectory_modulation(root_dir: str, session: int, da: float, params: dict, mode="Train"):

    n_trial = params["n_trajectory_trial"]
    simulation_time = params["train_simulation_time"]

    # make directory
    folder_name = root_dir + "trajectory_modulation/da%.2f/session%d" % (da, session)
    print(folder_name)
    os.makedirs(folder_name, exist_ok=True)

    # make environment
    env = Box()
    env.reward_range = 0.2
    env.goal[0] = 1.3
    env.goal[1] = 1.3

    # make network
    snn = EcHpcSimple(params={"conn_seed": session, "forced_run": True})
    goal_cnt = 0
    trial_end_time = np.ones(n_trial) * simulation_time

    # data array
    step_tot = 0
    place_tot = []
    spike_tot = []

    # virtual rat parameter
    v_rat = 0.2  # m/sec
    r = 0.15  # m
    period_1 = params["act_start"] * 0.001 + (1.3 - r - 0.75) / v_rat  # sec
    period_2 = r * 2 * np.pi * 0.25 / v_rat  # sec

    # return x_gain, y_gain
    def trajectory(t):
        if t < period_1:
            return 0, 1
        elif t < period_1 + period_2:
            l = (t - period_1) / v_rat
            theta = (l / (r * 2 * np.pi * 0.25)) * np.pi * 0.5
            return np.cos(90 - theta), np.sin(90 - theta)
        else:
            return 1, 0

    # trial loop
    for trial in range(n_trial):

        print("\nda %.2f, session %d, trial %d" % (da, session, trial))

        # make network
        simulator = cs.Simulator(snn, dt=params["dt"], max_spike_in_step=params["max_spike_in_step"], block_size=params["block_size"],
                                 wij_update_interval=params["wij_update_interval"], silent=True, method="rk4")
        simulator.snn.has_update_weight = False  # initially STDP is disabled
        simulator.write_to_var(var="f_ach", val=1.0)
        place = []
        rewarded_time = 10 ** 9

        # make agent
        rat = WaterMazeAgent(snn_n_dir=snn.n_dir, n_action=snn.n_action, seed=session * 1000 + trial)

        # initialize environment and agent
        rat.sp_prob = 0.0
        rat.set_velocity(0.0)
        rat.set_angle(np.pi * 0.5)
        observation = env.reset(ini_x=0.75, ini_y=0.75)  # state (x[m], y[m])

        # add initial input to the action cells
        simulator.add_electrode(snn.grid_1, t_start=100, t_end=200, array=rat.make_initial_input_to_grid(snn.grid_1, snn.ny))
        simulator.add_electrode(snn.grid_2, t_start=100, t_end=200, array=rat.make_initial_input_to_grid(snn.grid_2, snn.ny))
        # add electrode for the initial input to the grid cells
        init_dir = np.where(snn.action_dir == rat.angle)[0]
        ac_init = np.zeros(snn.action_neuron.n_cells)
        ac_init[init_dir] = 0.7
        ac_init[(init_dir + 1) % snn.n_action] = 0.5
        ac_init[init_dir - 1] = 0.5
        simulator.add_electrode(snn.action_neuron, t_start=params["act_start"] - 100, t_end=params["act_start"], array=ac_init)

        # simulation main loop
        progress = ProgressBar(min=0, max_value=simulation_time)
        for step in range(simulation_time):

            # update progressbar
            if step % 1000 == 0:
                progress.update(step)

            # start act
            if step == params["act_start"]:
                rat.set_velocity(params["rat_speed"])
                if mode == "Train":
                    simulator.snn.has_update_weight = True
                    simulator.write_to_var("trace", val=0.0)

            # the rat model starts exploration
            if step >= params["act_start"]:
                # get action
                action_spk_now = np.isin(snn.action_neuron.gid, simulator.get_spike_now()[:, 1])
                action = rat.get_action(env, action_spk_now)
                observation, reward, done = env.step(action, show=False)
                # update network
                if done and (rewarded_time > 10 ** 8):
                    simulator.add_to_var("da", 0.005, cell_group=snn.action_neuron)
                    rewarded_time = step
                    goal_cnt += 1

            # input for force running
            x_gain, y_gain = trajectory(step * 0.001)
            simulator.add_step_current(snn.action_neuron, array=rat.make_input_to_ac(0.15 - x_gain*0.03, 0.15 - y_gain*0.03))
            # update network
            simulator.add_step_current(snn.head_1, array=rat.make_input_to_hd(gain=params["g1_gain"]))
            simulator.add_step_current(snn.head_2, array=rat.make_input_to_hd(gain=params["g2_gain"]))
            simulator.step_advance(step)

            # write current place
            place.append(np.array([step * 0.001, observation.x, observation.y], dtype=np.float32))  # Time[s] X[m] Y[m]

            # end condition of the trial
            if step >= rewarded_time + rat.action_choice_interval or step == simulation_time - 1:
                # stack spike data
                spike = simulator.get_spike_data()
                spike[:, 0] += step_tot * 0.001
                spike_tot.append(spike)
                # stack place data
                place = np.vstack(place)
                place[:, 0] += step_tot * 0.001
                place_tot.append(place)
                # write trial time
                trial_end_time[trial] = step
                step_tot += step + 1
                break

        # close objects
        simulator.copy_wij_to_snn(snn)  # update snn wij
        del simulator

    # plot trajectory
    fig, ax = pyplot.subplots()
    ax.plot([0.0, 1.5, 1.5, 0.0, 0.0], [0.0, 0.0, 1.5, 1.5, 0.0], "-k")
    for trial, place in enumerate(place_tot):
        ax.plot(place[:, 1], place[:, 2], color=pyplot.cm.jet(trial / n_trial), alpha=0.3)
    ax.set_xlim([0, 1.5])
    ax.set_ylim([0, 1.5])
    pyplot.axis('equal')
    fig.savefig(folder_name + '/trajectory_da%.2f_session%d.png' % (da, session))
    pyplot.close(fig)

    # stack data of all sessions
    place_tot = np.vstack(place_tot)
    spike_tot = np.vstack(spike_tot)

    # save total log
    np.save(folder_name + '/spike_total_forced.npy', spike_tot)
    np.save(folder_name + '/place_total_forced.npy', place_tot)
    np.save(folder_name + "/trial_end_time_forced.npy", trial_end_time)

    # -------------------------- free exploration ------------------------- #

    # make environment
    env = Box()
    env.reward_range = 0.2
    env.goal[0] = 1.3
    env.goal[1] = 1.3
    goal_cnt = 0
    trial_end_time = np.ones(n_trial) * simulation_time

    # data array
    step_tot = 0
    place_tot = []
    spike_tot = []

    # trial loop
    for trial in range(n_trial):

        print("\nda %.2f, session %d, trial %d" % (da, session, trial))

        # make network
        simulator = cs.Simulator(snn, dt=params["dt"], max_spike_in_step=params["max_spike_in_step"], block_size=params["block_size"],
                                 wij_update_interval=params["wij_update_interval"], silent=True, method="rk4")
        simulator.snn.has_update_weight = False  # initially STDP is disabled
        simulator.write_to_var(var="f_ach", val=1.0)
        place = []
        rewarded_time = 10 ** 9

        # make agent
        rat = WaterMazeAgent(snn_n_dir=snn.n_dir, n_action=snn.n_action, seed=session * 1000 + trial)

        # initialize environment and agent
        rat.sp_prob = 0.0
        rat.set_velocity(0.0)
        rat.set_angle(np.pi * 0.5)
        observation = env.reset(ini_x=0.75, ini_y=0.75)  # state (x[m], y[m])

        # add electrode for the initial input to the grid cells
        simulator.add_electrode(snn.grid_1, t_start=100, t_end=200, array=rat.make_initial_input_to_grid(snn.grid_1, snn.ny))
        simulator.add_electrode(snn.grid_2, t_start=100, t_end=200, array=rat.make_initial_input_to_grid(snn.grid_2, snn.ny))
        # add initial input to the action cells
        init_dir = np.where(snn.action_dir == rat.angle)[0]
        ac_init = np.zeros(snn.action_neuron.n_cells)
        ac_init[init_dir] = 0.7
        ac_init[(init_dir + 1) % snn.n_action] = 0.5
        ac_init[init_dir - 1] = 0.5
        simulator.add_electrode(snn.action_neuron, t_start=params["act_start"]-100, t_end=params["act_start"], array=ac_init)

        # simulation main loop
        progress = ProgressBar(min=0, max_value=simulation_time)
        for step in range(simulation_time):

            # update progressbar
            if step % 1000 == 0:
                progress.update(step)

            # start act
            if step == params["act_start"]:
                rat.set_velocity(params["rat_speed"])
                if mode == "Train":
                    simulator.snn.has_update_weight = True
                simulator.write_to_var("trace", val=0.0)

            # the rat model starts exploration
            if step >= params["act_start"]:
                # get action
                action_spk_now = np.isin(snn.action_neuron.gid, simulator.get_spike_now()[:, 1])
                action = rat.get_action(env, action_spk_now)  # action (velocity[m/s], omega[rad/s])
                observation, reward, done = env.step(action, show=False)
                # update network
                if done and (rewarded_time > 10 ** 8):
                    simulator.add_to_var("da", 0.005, cell_group=snn.action_neuron)
                    simulator.add_to_var("da", da, cell_group=snn.pyramidal)
                    rewarded_time = step
                    goal_cnt += 1

            # update network
            simulator.add_step_current(snn.action_neuron, array=rat.make_input_to_ac(observation.x, observation.y))
            simulator.add_step_current(snn.head_1, array=rat.make_input_to_hd(gain=params["g1_gain"]))
            simulator.add_step_current(snn.head_2, array=rat.make_input_to_hd(gain=params["g2_gain"]))
            simulator.step_advance(step)

            # write current place
            place.append(np.array([step * 0.001, observation.x, observation.y], dtype=np.float32))  # Time[s] X[m] Y[m]

            # end condition of the trial
            if step >= rewarded_time + rat.action_choice_interval or step == simulation_time - 1:
                # stack spike data
                spike = simulator.get_spike_data()
                spike[:, 0] += step_tot * 0.001
                spike_tot.append(spike)
                # stack place data
                place = np.vstack(place)
                place[:, 0] += step_tot * 0.001
                place_tot.append(place)
                # write trial time
                trial_end_time[trial] = step
                step_tot += step + 1
                break

        # close objects
        simulator.copy_wij_to_snn(snn)  # update snn wij
        del simulator

    # plot trajectory
    fig, ax = pyplot.subplots()
    ax.plot([0.0, 1.5, 1.5, 0.0, 0.0], [0.0, 0.0, 1.5, 1.5, 0.0], "-k")
    for trial, place in enumerate(place_tot):
        ax.plot(place[:, 1], place[:, 2], color=pyplot.cm.jet(trial / n_trial), alpha=0.3)
    ax.set_xlim([0, 1.5])
    ax.set_ylim([0, 1.5])
    pyplot.axis('equal')
    fig.savefig(folder_name + '/trajectory_da%.2f_session%d_second.png' % (da, session))
    pyplot.close(fig)

    # stack data of all sessions
    place_tot = np.vstack(place_tot)
    spike_tot = np.vstack(spike_tot)

    # save total log
    np.save(folder_name + '/spike_total_free.npy', spike_tot)
    np.save(folder_name + '/place_total_free.npy', place_tot)
    np.save(folder_name + "/trial_end_time_free.npy", trial_end_time)
