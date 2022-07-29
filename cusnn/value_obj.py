import numpy as np
from typing import Optional, Union


class Variable:
    def __init__(self, name: str, dtype: str, init_val: Union[int, float], is_syn=False):
        self.name = name
        self.dtype = dtype
        self.ini_val = init_val
        self.is_syn = is_syn

    def __lt__(self, other):
        # self < other
        return self.name < other.name


class LocalVariable(Variable):
    def __init__(self, name: str, dtype: str, init_val: Union[int, float]):
        super().__init__(name, dtype, init_val)


class GlobalVariable(Variable):
    def __init__(self, name: str, dtype: str, init_val: Union[int, float], is_syn=False):
        super().__init__(name, dtype, init_val, is_syn)


class StateVariable(Variable):
    def __init__(self, name: str, dtype: str, init_val: Union[int, float], derivative: str):
        super().__init__(name, dtype, init_val)
        self.derivative = derivative
        self.token_list = None


class Constant:
    def __init__(self, name: str, dtype: str, array: np.ndarray):
        self.name = name
        self.array = array
        self.dtype = dtype


class Weight:
    def __init__(self, wij, wij_index):
        self.w = wij
        self.index = wij_index


class ConstCollector:
    def __init__(self, snn):
        self.__net_consts = []
        # default constants
        self.cell_type_index = np.zeros(snn.n_cells, dtype=np.int16)
        self.cell_type_width = np.zeros(snn.n_groups, dtype=np.int32)
        self.cell_type_edge = np.zeros(snn.n_groups, dtype=np.int32)
        cnt = 0
        for cell_type, group in enumerate(snn.cell_groups):
            self.cell_type_width[cell_type] = group.n_cells
            self.cell_type_edge[cell_type] = cnt
            self.cell_type_index[cnt:cnt + group.n_cells] = cell_type
            cnt += group.n_cells
        # if snn.n_cells <= 32000:
        #     self.__net_consts.append(Constant(name="CELL_TYPE_INDEX", dtype="short int", array=self.cell_type_index))
        self.__net_consts.append(Constant(name="CELL_TYPE_WIDTH", dtype="int", array=self.cell_type_width))
        self.__net_consts.append(Constant(name="CELL_TYPE_EDGE", dtype="int", array=self.cell_type_edge))
        self.__net_consts.append(Constant(name="WIJ_INDEX", dtype="int", array=snn.wij.index))

    @property
    def net_consts(self):
        return self.__net_consts
