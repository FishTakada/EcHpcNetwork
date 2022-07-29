import numpy as np
from typing import Union
from ._dataClass import SpikeData, TrajectoryData


__all__ = ["SpikeReader", "TrajectoryReader"]


# ----------------------------- Reader Class ----------------------------- #
class SpikeReader:
    def __init__(self, spike_data: np.ndarray):
        # self.__spikes = np.load(file_name)
        self.__spikes = spike_data  # [[spike_timing[sec], cell_id[#]],...[]]

    def make_group(self, member: list):
        ids = np.isin(self.__spikes[:, 1], member)
        return SpikeData(self.__spikes[ids, 0], self.__spikes[ids, 1], member)


class TrajectoryReader:
    def __init__(self, trajectory_data: np.ndarray, env_size: list):
        # self.__trajectory = np.load(file_name)
        self.__trajectory = trajectory_data
        self.__env_size = env_size

    def make_trajectory(self):
        return TrajectoryData(self.__trajectory[:, 0], self.__trajectory[:, 1], self.__trajectory[:, 2], env_size=self.__env_size)
