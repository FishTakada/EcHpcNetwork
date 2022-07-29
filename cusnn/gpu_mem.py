import pycuda.autoinit
import pycuda.driver as cuda

from typing import List, Union
import numpy as np
from .net_base import Module
from .value_obj import Weight


class GlobalMemoryVariable:
    def __init__(self, name: str, array: np.ndarray):
        self.name = name
        self.host = array
        self.device = cuda.to_device(self.host)

    def add(self, val, cell_group=None):
        cuda.memcpy_dtoh(self.host, self.device)
        if cell_group is None:
            self.host[:] += val
        else:
            self.host[cell_group.gid] += val
        cuda.memcpy_htod(self.device, self.host)

    def write(self, val, cell_group=None):
        cuda.memcpy_dtoh(self.host, self.device)
        if cell_group is None:
            self.host[:] = val
        else:
            self.host[cell_group.gid] = val
        cuda.memcpy_htod(self.device, self.host)

    def get(self, cell_group=None):
        cuda.memcpy_dtoh(self.host, self.device)
        if cell_group is None:
            return self.host
        else:
            return self.host[cell_group.gid]

    def dtoh(self):
        cuda.memcpy_dtoh(self.host, self.device)

    def htod(self):
        cuda.memcpy_htod(self.device, self.host)

    def __lt__(self, other):
        # self < other
        return self.name < other.name


class GpuMemory:
    def __init__(self, snn: Module, grid_size: int, block_size: int, gid2cid: np.ndarray):
        # make global memory variable list
        self.gm_list: List[GlobalMemoryVariable] = []
        self.__add_vars(snn)
        self.gid2cid = self.add_gm('gid2cid', gid2cid)
        self.i_ext = self.add_gm('i_ext_gm', np.zeros(snn.n_cells, dtype=np.float32))
        self.wij = self.add_gm('wij', snn.wij.w)
        self.spike_log = self.add_gm('spike_log', np.ones(grid_size * block_size, dtype=np.int32) * -1)
        self.spike_log_past = self.add_gm('spike_log_past', np.ones(grid_size * block_size, dtype=np.int32) * -1)

        # make arg list for kernel function
        self.gm_list.sort()
        self.gm_args = [gm.device for gm in self.gm_list]

    def __add_vars(self, snn: Module):
        gpu_set = set()
        for cg in snn.cell_groups:
            for var in cg.g_vars:
                if var.is_syn:
                    gpu_set.add((var.name, var.dtype, "syn", var.ini_val))
                else:
                    gpu_set.add((var.name, var.dtype, "cell", var.ini_val))
        for name, dtype, cors, ini_val in gpu_set:
            if cors == "cell":
                self.add_gm(name, ini_val * np.ones(snn.n_cells).astype(self.str2dtype(dtype)))
            if cors == "syn":
                self.add_gm(name, ini_val * np.ones(snn.n_cells ** 2).astype(self.str2dtype(dtype)))

    def add_gm(self, var_name: str, array: np.ndarray) -> GlobalMemoryVariable:
        if array.size != 0:     # if no synapse
            self.gm_list.append(GlobalMemoryVariable(var_name, array))
        else:
            self.gm_list.append(GlobalMemoryVariable(var_name, np.zeros(1, dtype=np.float32)))
        return self.gm_list[-1]

    @staticmethod
    def str2dtype(dtype: str):
        if dtype == "float":
            return np.float32
        elif dtype == "int":
            return np.int32
