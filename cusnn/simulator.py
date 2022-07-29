import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

import numpy as np
from .net_base import Module
from .cu_make import DeviceFuncMaker
from .gpu_mem import GpuMemory, GlobalMemoryVariable
from .value_obj import ConstCollector, Variable
from .electrode import Electrode
from typing import Union


__all__ = ["Simulator"]


class Simulator:
    def __init__(self, snn: Module, **params):

        self.snn = snn
        # simulation parameter
        self.gm_swap = params.get("gm_swap", True)
        self.method = params.get("method", "euler")
        self.dt = params.get("dt", 0.2)  # ms
        self.syn_delay = params.get("syn_delay", 1.0)  # ms
        self.max_spike_in_step = params.get("max_spike_in_step", 1024)
        self.block_size = params.get("block_size", 64)
        self.wij_update_interval = params.get("wij_update_interval", 20)

        # make gid to cid(cell id) index array
        # self.gid2cid = np.array([], dtype=np.int32)
        # self.gid2cid = np.hstack((self.gid2cid, np.arange(snn.n_cells, dtype=np.int32)))
        self.gid2cid = np.arange(snn.n_cells, dtype=np.int32)
        rem = (self.block_size - (snn.n_cells % self.block_size)) % self.block_size
        self.gid2cid = np.hstack((self.gid2cid, np.ones(rem, dtype=np.int32) * -1))
        self.grid_size = int(self.gid2cid.size / self.block_size)

        # show network size
        if not params.get("silent", False):
            print("number of cells: %d" % snn.n_cells)
            print("number of cell groups: %d" % snn.n_groups)
            print("number of synapse: %d" % snn.wij.w.size)
            print("synaptic weight size: %.3f MB" % (snn.wij.w.size * 4 / 1000000))
            print("total constant memory usage: %d Byte" % (8*snn.n_groups + 4*snn.n_groups*snn.n_groups))
            print("Grid size: %d" % self.grid_size)
            print("Block size: %d" % self.block_size)

        # make objects
        self.spike_data = []
        self.electrode_list = []
        self.step_current_list = []
        self.spike_now = np.empty((0, 2), dtype=np.int)
        self.cc = ConstCollector(snn)

        # allocate global memory
        self.gpu_mem = GpuMemory(self.snn, self.grid_size, self.block_size, self.gid2cid)
        self.cm = DeviceFuncMaker(self, self.cc, snn, method=self.method, gm_swap=self.gm_swap)
        if params.get("save_cu_code", False):
            # print("Global memory order check:")
            # for gm in self.gpu_mem.gm_list:
            #     print("%s %s" % (gm.host.dtype, gm.name))
            with open(snn.__class__.__name__ + ".cu", "w") as f:
                f.write(self.cm.cu_code)

        # compile .cu code then get kernel function
        if not params.get("silent", False):
            self.mod = SourceModule(self.cm.cu_code, no_extern_c=True)
        else:
            self.mod = SourceModule(self.cm.cu_code, options=["-w"], no_extern_c=True)
        self.integration = self.mod.get_function("step_advance")
        if snn.has_update_weight:
            self.__update_weight = self.mod.get_function("update_weight")
        # set value of constant memory
        for constant in self.cc.net_consts:
            constant_mem = self.mod.get_global(constant.name)[0]
            cuda.memcpy_htod(constant_mem, constant.array)

    def step_advance(self, n_step):
        # update i_ext_hm
        self._update_i_ext_hm(n_step)
        # numerical integration for synaptic delay
        self.integration(*self.gpu_mem.gm_args, block=(self.block_size, 1, 1), grid=(self.grid_size, 1, 1))
        if self.snn.has_update_weight and n_step % self.wij_update_interval == 0:
            self.__update_weight(*self.gpu_mem.gm_args, block=(self.block_size, 1, 1), grid=(self.grid_size, 1, 1))
        # copy spikes to host memory
        self.gpu_mem.spike_log.dtoh()
        # spike_log_past(n+1) <- spike_log(n)
        cuda.memcpy_dtod(self.gpu_mem.spike_log_past.device, self.gpu_mem.spike_log.device, self.gpu_mem.spike_log.host.nbytes)
        # find spikes
        spike_index = np.where(self.gpu_mem.spike_log.host != -1)[0]
        # write spikes
        if len(spike_index):
            self.spike_now = np.array([(self.gpu_mem.spike_log.host[spike_index] * self.dt + n_step * self.syn_delay) * 0.001, spike_index],
                                      dtype=np.float32).transpose()
            self.spike_data.append(self.spike_now)

    def step_advance_no_output(self, n_step):
        # update i_ext_hm
        self._update_i_ext_hm(n_step)
        # numerical integration for synaptic delay
        self.integration(*self.gpu_mem.gm_args, block=(self.block_size, 1, 1), grid=(self.grid_size, 1, 1))
        if self.snn.has_update_weight and n_step % self.wij_update_interval == 0:
            self.__update_weight(*self.gpu_mem.gm_args, block=(self.block_size, 1, 1), grid=(self.grid_size, 1, 1))
        # spike_log_past(n+1) <- spike_log(n)
        cuda.memcpy_dtod(self.gpu_mem.spike_log_past.device, self.gpu_mem.spike_log.device, self.gpu_mem.spike_log.host.nbytes)

    ''' utils '''
    def get_cuda_code(self):
        return self.cm.cu_code

    def copy_wij_to_snn(self, snn: Module):
        cuda.memcpy_dtoh(snn.wij.w, self.gpu_mem.wij.device)

    ''' current injection '''
    def add_electrode(self, cell_group, t_start, t_end, array: np.ndarray):
        self.electrode_list.append(Electrode(cell_group, t_start, t_end, array))

    def add_step_current(self, cell_group, array: np.ndarray):
        self.gpu_mem.i_ext.host[cell_group.gid] += array

    ''' read and write global memory variable '''
    def get_wij(self, pre_group, post_group) -> np.ndarray:
        # get weight of pyramidal to action
        wij_ind = self.snn.wij.index[pre_group.cell_type, post_group.cell_type]
        return self.gpu_mem.wij.get()[wij_ind:(wij_ind + pre_group.n_cells * post_group.n_cells)].reshape((pre_group.n_cells, post_group.n_cells))

    def write_wij(self, pre_group, post_group, array: np.ndarray):
        wij_ind = self.snn.wij.index[pre_group.cell_type, post_group.cell_type]
        self.gpu_mem.wij.dtoh()
        self.gpu_mem.wij.host[wij_ind:(wij_ind + pre_group.n_cells * post_group.n_cells)] = array.flatten()
        self.gpu_mem.wij.htod()

    def add_to_var(self, var: Union[Variable, str], val, cell_group=None):
        self.find_gpu_var(var).add(val, cell_group)

    def write_to_var(self, var: Union[Variable, str], val, cell_group=None):
        self.find_gpu_var(var).write(val, cell_group)

    def get_var(self, var: Union[Variable, str], cell_group=None):
        return self.find_gpu_var(var).get(cell_group)

    def find_gpu_var(self, var: Union[Variable, str]) -> GlobalMemoryVariable:
        if isinstance(var, Variable):
            var_name = var.name
        else:
            var_name = var
        for i, gpu_var in enumerate(self.gpu_mem.gm_list):
            if var_name == gpu_var.name:
                return self.gpu_mem.gm_list[i]
        print("var '%s' was not found in this SNN model" % var_name)
        exit()

    ''' get spike data '''
    # [[spike_timing[sec], cell_id[#]],...[]]
    def get_spike_data(self):
        if len(self.spike_data) != 0:
            return np.vstack(self.spike_data)
        else:
            print("Nothing was fired.")

    # [[spike_timing[sec], cell_id[#]],...[]]
    def save_spike_data(self, file_name: str):
        if len(self.spike_data) != 0:
            np.save(file_name, self.get_spike_data())
        else:
            print("Nothing was fired.")

    def get_spike_now(self):
        return self.spike_now

    ''' private funcs '''
    def _update_i_ext_hm(self, n_step):
        self.electrode_list = [ele for ele in self.electrode_list if n_step < ele.t_end]
        for ele in self.electrode_list:
            if ele.t_start <= n_step:
                self.gpu_mem.i_ext.host[ele.gid] += ele.array  # add current
        cuda.memcpy_htod(self.gpu_mem.i_ext.device, self.gpu_mem.i_ext.host)
        self.gpu_mem.i_ext.host *= .0   # refresh i_ext

