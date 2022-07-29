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
    metadata = {'render.modes': ['human', 'rgb_array']}

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
        self.robot_radius = 0.05    # [m]
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
        # self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # set action_space (velocity[m/s], yaw[rad])
        act_low = np.float32(np.array([min_linear_velocity, min_angle]))
        act_high = np.float32(np.array([max_linear_velocity, max_angle]))
        self.action_space = spaces.Box(act_low, act_high)

        # initialize state (x[m], y[m])
        self.__state = self.reset()
        self.goal = State(1.35, 1.35)  # goal [x(m), y(m)]
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
        self.__state = State(ini_x, ini_y)  # np.array(ini_state)
        return self.state

    def collision_made_by(self, action):
        nx = self.state.x + action[0] * math.cos(action[1]) * self.dt
        ny = self.state.y + action[0] * math.sin(action[1]) * self.dt
        return self.__is_collision(nx, ny)

    def render(self, mode='human', close=False):
        scale_width = self.screen_width / float(self.x_size)
        scale_height = self.screen_height / float(self.y_size)

        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(self.screen_width, self.screen_height)
            # wall
            wall = rendering.make_capsule(self.screen_width, 4)
            wall_trans = rendering.Transform()
            wall.add_attr(wall_trans)
            wall.set_color(0.2, 0.4, 1.0)
            wall_trans.set_translation(0, 0)
            wall_trans.set_rotation(0)
            self.viewer.add_geom(wall)

            wall = rendering.make_capsule(self.screen_width, 4)
            wall_trans = rendering.Transform()
            wall.add_attr(wall_trans)
            wall.set_color(0.2, 0.4, 1.0)
            wall_trans.set_translation(0, 0)
            wall_trans.set_rotation(math.pi / 2)
            self.viewer.add_geom(wall)

            wall = rendering.make_capsule(self.screen_width, 4)
            wall_trans = rendering.Transform()
            wall.add_attr(wall_trans)
            wall.set_color(0.2, 0.4, 1.0)
            wall_trans.set_translation(0, self.screen_height)
            wall_trans.set_rotation(0)
            self.viewer.add_geom(wall)

            wall = rendering.make_capsule(self.screen_width, 4)
            wall_trans = rendering.Transform()
            wall.add_attr(wall_trans)
            wall.set_color(0.2, 0.4, 1.0)
            wall_trans.set_translation(self.screen_width, 0)
            wall_trans.set_rotation(math.pi / 2)
            self.viewer.add_geom(wall)

            # start
            start = rendering.make_circle(self.robot_radius / self.xy_resolution * scale_width)
            start_trans = rendering.Transform()
            start.add_attr(start_trans)
            start.set_color(0.5, 1.0, 0.5)
            start_trans.set_translation(self.state.x / self.xy_resolution * scale_width,
                                        self.state.y / self.xy_resolution * scale_height)
            self.viewer.add_geom(start)

            # goal
            goal = rendering.make_circle(self.reward_range / self.xy_resolution * scale_width)
            goal_trans = rendering.Transform()
            goal.add_attr(goal_trans)
            goal.set_color(1.0, 0.0, 0.0)
            goal_trans.set_translation(self.goal[0] / self.xy_resolution * scale_width,
                                       self.goal[1] / self.xy_resolution * scale_height)
            self.viewer.add_geom(goal)

            # robot pose
            robot = rendering.make_circle(self.robot_radius / self.xy_resolution * scale_width)
            self.robot_trans = rendering.Transform()
            robot.add_attr(self.robot_trans)
            robot.set_color(0.0, 0.0, 1.0)
            self.viewer.add_geom(robot)

            # robot yaw rate
            orientation = rendering.make_capsule(self.robot_radius / self.xy_resolution * scale_width, 2.0)
            self.orientation_trans = rendering.Transform()
            orientation.add_attr(self.orientation_trans)
            orientation.set_color(0.0, 1.0, 0.0)
            self.viewer.add_geom(orientation)

        robot_x = self.state.x / self.xy_resolution * scale_width
        robot_y = self.state.y / self.xy_resolution * scale_height

        self.robot_trans.set_translation(robot_x, robot_y)
        self.orientation_trans.set_translation(robot_x, robot_y)
        self.orientation_trans.set_rotation(self.state[2])

        robot = rendering.make_circle(self.robot_radius / self.xy_resolution * scale_width)
        self.robot_trans = rendering.Transform()
        robot.add_attr(self.robot_trans)
        robot.set_color(0.0, 0.0, 1.0)
        self.robot_trans.set_translation(robot_x, robot_y)
        self.viewer.add_onetime(robot)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

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
        if math.sqrt((self.state.x - self.goal.x) ** 2 + (self.state.y - self.goal.y) ** 2) <= self.reward_range:
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
