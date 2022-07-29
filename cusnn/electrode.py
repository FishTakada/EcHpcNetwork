import numpy as np
from .cells import CellGroup


__all__ = ["Electrode"]


class Electrode:
    def __init__(self, cell_group: CellGroup, t_start, t_end, array: np.ndarray):
        self.__gid = cell_group.gid
        self.__t_start = t_start
        self.__t_end = t_end
        self.__array = array

    @property
    def t_start(self):
        return self.__t_start

    @property
    def t_end(self):
        return self.__t_end

    @property
    def gid(self):
        return self.__gid

    @property
    def array(self):
        return self.__array
