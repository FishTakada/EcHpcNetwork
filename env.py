import numpy as np
import math
import gym
from gym import spaces
from gym.utils import seeding


class State:
    def __init__(self, x, y):
        self.x = x  # [m]
        self.y = y  # [m]


class Box(gym.Env):
    def __init__(self, x_max=1.5, y_max=1.5):

        # world param
        self.dt = 0.001     # [sec]
        self.xy_resolution = 0.025   # [m]
        self.x_max, self.y_max = x_max, y_max   # [m]
        self.x_size = int(self.x_max / self.xy_resolution)
        self.y_size = int(self.y_max / self.xy_resolution)
        self.screen_width = 500     # [pixel]
        self.screen_height = 500       # [pixel]

        # robot param
        self.robot_radius = 0.01    # [m]
        self.reward_range = 0.15    # [m]

        # action param
        max_linear_velocity = 0.2  # [m/s]
        min_linear_velocity = 0.0  # [m/s]
        min_angle = 0
        max_angle = 2 * math.pi

        # set observation space (x[m], y[m], yaw[rad], v[m/s]])
        obs_low = np.float32(np.array([0., 0., 0., min_linear_velocity]))
        obs_high = np.float32(np.array([self.x_max, self.y_max, max_angle, max_linear_velocity]))
        self.observation_space = spaces.Box(obs_low, obs_high)

        # set action_space (velocity[m/s], yaw[rad])
        act_low = np.float32(np.array([min_linear_velocity, min_angle]))
        act_high = np.float32(np.array([max_linear_velocity, max_angle]))
        self.action_space = spaces.Box(act_low, act_high)

        # initialize state (x[m], y[m])
        self.__state: State = self.reset()

        self.goal = np.array([1.35, 1.35])  # goal [x(m), y(m)]
        self.area = self.__initialize_area()
        self.obstacle = np.where(0 < self.area)

        # viewer handles
        self.viewer = None
        self.robot_trans = None
        self.orientation_trans = None

    @property
    def state(self):
        return self.__state

    def seed(self, seed=None):
        _, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: np.ndarray, show=True):
        self.__update_state(action)
        reward = self.__get_reward()
        return self.state, reward, self.__is_done(show)

    def reset(self, ini_x=0.75, ini_y=0.75):
        # state (x[m], y[m])
        self.__state = State(ini_x, ini_y)
        return self.state

    def collision_made_by(self, action):
        nx = self.state.x + action[0] * math.cos(action[1]) * self.dt
        ny = self.state.y + action[0] * math.sin(action[1]) * self.dt
        return self.__is_collision(nx, ny)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    # ----------------------- private funcs ------------------------- #
    def __update_state(self, action):
        self.__state.x += action[0] * math.cos(action[1]) * self.dt  # x[m]
        self.__state.y += action[0] * math.sin(action[1]) * self.dt  # y[m]

    def __initialize_area(self):
        grid_area = np.zeros((self.y_size, self.x_size), dtype=np.int32)
        grid_area[:, 0] = 1
        grid_area[:, -1] = 1
        grid_area[0, :] = 1
        grid_area[-1, :] = 1
        return grid_area

    def __get_reward(self):
        if self.__is_goal():
            return 3.0
        elif self.__is_collision(self.state.x, self.state.y):
            return -5
        else:
            return -1

    def __is_done(self, show=False):
        return self.__is_collision(self.state.x, self.state.y, show) or self.__is_goal(show)

    def __is_goal(self, show=False):
        if math.sqrt((self.state.x - self.goal[0]) ** 2 + (self.state.y - self.goal[1]) ** 2) <= self.reward_range:
            if show:
                print("Goal")
            return True
        else:
            return False

    def __is_collision(self, x, y, show=False):
        x, y = self.__to_xy_index(x, y)
        if np.sum((np.sqrt(np.power(x - self.obstacle[1], 2) + np.power(y - self.obstacle[0], 2)) * self.xy_resolution) < self.robot_radius):
            if show:
                print("collision")
            return True
        return False

    def __to_xy_index(self, x, y):
        return int(x / self.xy_resolution), int(y / self.xy_resolution)
