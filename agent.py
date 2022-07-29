import numpy as np
from env import Box


pi2 = np.pi * 2


class Agent:
    def __init__(self, n_dir, n_sp, min_sp=0.0, max_sp=0.4):
        # parameter for a virtual rat agent
        self.n_dir = n_dir
        self.min_velocity, self.max_velocity = min_sp, max_sp  # min and max velocity of agent [m/s]
        self.angle_space = np.array([(2 * np.pi / n_dir * i) for i in range(n_dir)])  # [rad]
        self.velocity_space = np.linspace(self.min_velocity, self.max_velocity, n_sp)  # [m/s]

        # initialize state
        self.state = np.zeros(2, np.int)  # index of list [velocity, angle]

    @property
    def velocity(self):
        return self.velocity_space[self.state[0]]

    @property
    def angle(self):
        return self.angle_space[self.state[1]]

    # set rat speed [m/s]
    def set_velocity(self, velocity):
        self.state[0] = np.searchsorted(self.velocity_space, velocity, side="left")

    # set rat head direction [rad]
    def set_angle(self, angle):
        self.state[1] = np.searchsorted(self.angle_space, angle, side="left")


class LinearTrackAgent(Agent):
    def __init__(self, snn_n_dir, n_dir, n_sp=150):

        super().__init__(n_dir, n_sp, min_sp=0.0, max_sp=0.4)

        # initialize state variables
        self.state = np.array([int(n_sp / 2), 0])  # index of list [velocity, angle]
        self.snn_angle_space = np.array([(2 * np.pi / snn_n_dir * i) for i in range(snn_n_dir)])  # [rad]
        self.vx, self.vy = np.cos(self.snn_angle_space), np.sin(self.snn_angle_space)

    def get_action(self, env_obj):
        # check boundary condition
        act = np.array([self.velocity, self.angle])
        while env_obj.collision_made_by(act):
            self.state[1] = (self.state[1] + 1) % self.n_dir  # turn right
            act = np.array([self.velocity, self.angle])
        return act

    def make_initial_input_to_grid(self, cell_group, ny, center_id=None):
        i_ext = np.zeros(cell_group.n_cells, dtype=np.float32)
        if center_id is not None:
            inj_pos = np.array([center_id - ny, center_id - 1, center_id, center_id + 1, center_id + ny - 1, center_id + ny, center_id + ny + 1],
                               dtype=np.int32)
        else:
            inj_pos = np.array([1, ny, ny + 1, ny + 2, ny * 2, ny * 2 + 1, ny * 2 + 2], dtype=np.int32)
        i_ext[inj_pos] = 3.0
        return i_ext

    def make_input_to_speed(self, cell_group, stim_min, stim_max):
        speed_stim = (stim_max - stim_min) * (self.velocity - self.min_velocity) / (
                self.max_velocity - self.min_velocity) + stim_min
        return np.ones(cell_group.n_cells, dtype=np.float32) * speed_stim

    def make_input_to_hd(self, gain):
        return (self.vx * np.cos(self.angle) + self.vy * np.sin(self.angle)).astype(np.float32) * self.velocity * gain + 0.85


class WaterMazeAgent:
    def __init__(self, snn_n_dir, n_action, seed):

        self.n_action = n_action
        self.rs = np.random.RandomState(seed)

        # parameter for agent
        self.hd_prob = 0.2
        self.hd_diff = pi2 / 16

        # state variable
        self.velocity = 0.2     # [m/s]
        self.angle = self.rs.random_sample() * np.pi * 2  # [m/s]

        self.snn_angle_space = np.array([(2 * np.pi / snn_n_dir * i) for i in range(snn_n_dir)])  # [rad]
        self.vx, self.vy = np.cos(self.snn_angle_space), np.sin(self.snn_angle_space)

        self.action_x = np.cos([(pi2 / n_action * i) for i in range(n_action)])
        self.action_y = np.sin([(pi2 / n_action * i) for i in range(n_action)])

        # stim parameters
        self.step_cnt = 0
        self.action_spike_cnt = np.zeros(self.n_action)
        self.action_choice_interval = 100  # [ms]
        self.initial_input_to_grid = 3.0
        self.wall_gain = 1.5

    def set_velocity(self, velocity):
        self.velocity = velocity

    def set_angle(self, angle):
        self.angle = angle

    def get_action(self, env_obj: Box, action_spike):

        # count spikes of action neuron
        self.action_spike_cnt[action_spike] += 1
        self.step_cnt += 1

        # update head direction
        if self.step_cnt % self.action_choice_interval == 0:
            self.step_cnt = 0
            # state transition
            if np.sum(self.action_spike_cnt):
                # mean vec
                dx = np.mean(self.action_spike_cnt * self.action_x)
                dy = np.mean(self.action_spike_cnt * self.action_y)
                self.angle = (np.arctan2(dy, dx) + pi2) % pi2

                # reset spike
                self.action_spike_cnt = np.zeros(self.n_action)
            else:
                # state transition (determined probabilistically if ac cells did not fire)
                val = self.rs.random_sample()
                if val < self.hd_prob:
                    self.angle = (self.angle + self.hd_diff) % pi2  # turn left
                elif val < self.hd_prob * 2:
                    self.angle = (self.angle - self.hd_diff + pi2) % pi2  # turn right

        # check boundary condition
        act = np.array([self.velocity, self.angle])
        if self.rs.random_sample() < 0.5:
            while env_obj.collision_made_by(act):
                self.angle = (self.angle + self.hd_diff) % pi2  # turn left
                act = np.array([self.velocity, self.angle])
        else:
            while env_obj.collision_made_by(act):
                self.angle = (self.angle - self.hd_diff + pi2) % pi2  # turn right
                act = np.array([self.velocity, self.angle])

        return act

    def make_initial_input_to_grid(self, cell_group, ny, center_id=None):
        i_ext = np.zeros(cell_group.n_cells, dtype=np.float32)
        if center_id is not None:
            inj_pos = np.array([center_id - ny, center_id - 1, center_id, center_id + 1, center_id + ny - 1, center_id + ny, center_id + ny + 1],
                               dtype=np.int32)
        else:
            inj_pos = np.array([1, ny, ny + 1, ny + 2, ny * 2, ny * 2 + 1, ny * 2 + 2], dtype=np.int32)
        i_ext[inj_pos] = self.initial_input_to_grid
        return i_ext

    def make_input_to_ac(self, x, y):
        return (self.action_x * (max(0, 0.15 - x) - max(0, x - 1.35)) + self.action_y * (max(0, 0.15 - y) - max(0, y - 1.35))) * self.wall_gain

    def make_input_to_hd(self, gain):
        return (self.vx * np.cos(self.angle) + self.vy * np.sin(self.angle)).astype(np.float32) * self.velocity * gain + 0.85


class FreeExplorationAgent:
    def __init__(self, snn_n_dir, n_dir, seed):

        # parameter for agent
        self.hd_prob = 0.01  # transition probability of the head angle state
        self.sp_prob = 0.0005  # transition probability of the velocity state
        self.hd_diff = pi2 / n_dir

        # initialize state variables
        self.rs = np.random.RandomState(seed)

        # state variable
        self.velocity = 0.2     # [m/s]
        self.angle = self.rs.random_sample() * np.pi * 2  # [m/s]

        self.snn_angle_space = np.array([(2 * np.pi / snn_n_dir * i) for i in range(snn_n_dir)])  # [rad]
        self.vx, self.vy = np.cos(self.snn_angle_space), np.sin(self.snn_angle_space)

        # stim parameters
        self.initial_input_to_grid = 3.0
        self.sp_stim_min, self.sp_stim_max = 0, 5
        self.wall_gain = 0.1

    def set_velocity(self, velocity):
        self.velocity = velocity

    def set_angle(self, angle):
        self.angle = angle

    def get_action(self, env_obj: Box):

        # state transition (head angle)
        val = self.rs.random_sample()
        if val < self.hd_prob:
            self.angle = (self.angle + self.hd_diff) % pi2  # turn left
        elif val < self.hd_prob * 2:
            self.angle = (self.angle - self.hd_diff + pi2) % pi2  # turn right

        dx = np.cos(self.angle) + (max(0, 0.1 - env_obj.state.x) - max(0, env_obj.state.x - 1.4)) * self.wall_gain
        dy = np.sin(self.angle) + (max(0, 0.1 - env_obj.state.y) - max(0, env_obj.state.y - 1.4)) * self.wall_gain
        self.angle = (np.arctan2(dy, dx) + pi2) % pi2

        # check boundary condition
        act = np.array([self.velocity, self.angle])
        if self.rs.random_sample() < 0.5:
            while env_obj.collision_made_by(act):
                self.angle = (self.angle + self.hd_diff) % pi2  # turn left
                act = np.array([self.velocity, self.angle])
        else:
            while env_obj.collision_made_by(act):
                self.angle = (self.angle - self.hd_diff + pi2) % pi2  # turn right
                act = np.array([self.velocity, self.angle])

        return act

    def make_initial_input_to_grid(self, cell_group, ny, center_id=None):
        i_ext = np.zeros(cell_group.n_cells, dtype=np.float32)
        if center_id is not None:
            inj_pos = np.array([center_id - ny, center_id - 1, center_id, center_id + 1, center_id + ny - 1, center_id + ny, center_id + ny + 1],
                               dtype=np.int32)
        else:
            inj_pos = np.array([1, ny, ny + 1, ny + 2, ny * 2, ny * 2 + 1, ny * 2 + 2], dtype=np.int32)
        i_ext[inj_pos] = self.initial_input_to_grid
        return i_ext

    def make_input_to_hd(self, gain):
        return (self.vx * np.cos(self.angle) + self.vy * np.sin(self.angle)).astype(np.float32) * self.velocity * gain + 0.85