from .value_obj import ConstCollector
from .net_base import Module
from .cells import CellGroup
from .tokenizer import euler_method, rk4_method, stdp_swap
from typing import Union


class DeviceFuncMaker:
    def __init__(self, simulator, cc: ConstCollector, snn: Module, method: str, gm_swap: bool):
        self.__cu_code = ""
        self.__base = """
$constant_memory$
extern "C"{
    $cid2cell_type$
    __global__ void step_advance($args$)
    {
        const int tid = threadIdx.x;    // thread id
        const int gid = blockDim.x * blockIdx.x + tid;  // global id
        //const int cid = gid2cid[gid];  // cell id
        const int cid = gid;  // cell id
        const int post_cti = CELL_TYPE_INDEX[cid];
        const int post_ctw = CELL_TYPE_WIDTH[post_cti];
        const int post_iid = (cid - CELL_TYPE_EDGE[post_cti]); // id inside small wij
        
        // allocate vars
        int i, count, pre_cti, pre_cid, d_step, spike_total, wij_idx;
        $allocate_var$
        
        // allocate shared memory
        __shared__ int spike_log_sm[$MAX_SPIKE_IN_STEP$][2];    // [id, n_step]...
        __shared__ int spike_num_sm[$BLOCK_SIZE$];
        __shared__ int spike_total_sm;
        
        // calc spike data for next step
        spike_log[cid] = -1;    // initialize spike log
        __syncthreads();
        
        count = 0;
        for(i=0; i<$GRID_SIZE$; ++i){
            if(spike_log_past[$GRID_SIZE$ * tid + i] != -1)count++;
        }
        spike_num_sm[tid] = count;
        __syncthreads();
        if(tid == 0){
            for(i=1; i<$BLOCK_SIZE$; ++i){
                spike_num_sm[i] = spike_num_sm[i] + spike_num_sm[i-1];
            }
        }
        __syncthreads();
        count = 0;
        spike_total = spike_num_sm[$BLOCK_SIZE$-1];
        for(i=$GRID_SIZE$-1; i>=0; --i){
            if(spike_log_past[$GRID_SIZE$ * tid + i] != -1){
                spike_log_sm[spike_num_sm[tid] - count - 1][0] = $GRID_SIZE$ * tid + i;
                spike_log_sm[spike_num_sm[tid] - count - 1][1] = spike_log_past[$GRID_SIZE$ * tid + i];
                count += 1;
            }
        }
        __syncthreads();
        
        //if(cid != -1){
$calc_group$
        //}
    }
$update_weight$
}  
"""
        self.calc_group = ""
        self.calc_group_temp = """
            $pre_loop$
            i_ext = i_ext_gm[cid];
            //__syncthreads();
            // dt loop
            for(d_step=0; d_step<$DELAY_STEP$; d_step++){
                $spike_receive$
                $pre_update$
                $update$
                if($spike_detection$){
                    $spike_event$
                }
            }
            $post_loop$
        """
        self.syn_event_tmp = """
                i_syn = 0.0f;
                for(i=0; i<spike_total; i++){
                    if(spike_log_sm[i][1] == d_step){
                        pre_cid = spike_log_sm[i][0];
                        pre_cti = CELL_TYPE_INDEX[pre_cid];
                        wij_idx = WIJ_INDEX[$NUM_CELL_TYPE$ * pre_cti + post_cti];
                        if (wij_idx != -1){
                            wij_pre_post = wij_idx + (pre_cid - CELL_TYPE_EDGE[pre_cti]) * post_ctw + post_iid;
                            g_syn = wij[wij_pre_post];
                            $spike_receive$
                            $stdp_pre_post$
                        }
                        $stdp_post_pre$
                    }
                }
        """

        self.update_weight = """
    __global__ void update_weight($args$)
    {
        // calc id
        const int tid = threadIdx.x;    // thread id
        const int gid = blockDim.x * blockIdx.x + tid;  // global id
        const int cid = gid2cid[gid];  // global id
        const int post_cti = CELL_TYPE_INDEX[cid];
        const int post_ctw = CELL_TYPE_WIDTH[post_cti];
        const int post_iid = (cid - CELL_TYPE_EDGE[post_cti]); // id inside small wij

        // variables
        float syn;
        int wij_pre_post, pre_cti;
        
        if(cid != -1){
            for(pre_cti=0; pre_cti<49; pre_cti++){
                $w_update$
            }
        }
    }"""

        self.constant_memory = "// constant memory"
        self.allocate_var = set()

        self.s_var_name = set()
        for cg in snn.cell_groups:
            for var in cg.s_vars:
                self.s_var_name.add(var.name)

        # ------------------ make arg of the kernel function ------------------ #
        self.gpu_args = []
        arg_set = {("i_ext_gm", "float"), ("gid2cid", "int"), ("wij", "float"), ("spike_log", "int"), ("spike_log_past", "int")}
        for cg in snn.cell_groups:
            for var in cg.g_vars:
                arg_set.add((var.name, var.dtype))
        for name, dtype in sorted(list(arg_set)):
            self.gpu_args.append("%s *%s" % (dtype, name))

        # ------------------ add update weight to base code ------------------ #
        if snn.has_update_weight:
            self.__base = self.__base.replace("$update_weight$", self.update_weight)
        else:
            self.__base = self.__base.replace("$update_weight$", "")

        # ------------------  ------------------ #
        cnt = 0
        if isinstance(snn, Module):
            for i, cg in enumerate(snn.calc_groups):
                # update_func = self.__add_update_func(cg, method, gm_swap)
                # rem = (simulator.block_size - (snn.calc_group_n_cells[i] % simulator.block_size)) % simulator.block_size
                # self.calc_group += "\t\tif(%d <= gid && gid < %s){" % (cnt, cnt + snn.calc_group_n_cells[i] + rem)
                # self.calc_group += "\n\t\t\t%s" % update_func
                # self.calc_group += "\n\t\t}\n"
                # cnt += snn.calc_group_n_cells[i] + rem
                update_func = self.__add_update_func(cg, method, gm_swap)
                if i == 0:
                    self.calc_group += "\t\tif(%d <= cid && cid < %s){" % (cnt, cnt + snn.calc_group_n_cells[i])
                else:
                    self.calc_group += "\t\telse if(%d <= cid && cid < %s){" % (cnt, cnt + snn.calc_group_n_cells[i])
                self.calc_group += "\n\t\t\t%s" % update_func
                self.calc_group += "\n\t\t}\n"
                cnt += snn.calc_group_n_cells[i]

        for constant in cc.net_consts:
            self.constant_memory += "\n__device__ __constant__ %s %s[%d];" % (constant.dtype, constant.name, constant.array.size)

        # make
        self.__make_cu_code()
        self.__insert_local_variable()
        # replace
        self.__replace_stdp(snn, gm_swap)
        self.__replace_simulation_params(simulator)

        # if snn.n_cells > 32000:
        if True:
            self.cid2cell_type = """__device__ int cid2cell_type(int cid){\n"""
            for i, edge in enumerate(cc.cell_type_edge[1:]):
                self.cid2cell_type += "\t\tif(cid < %d)return %d;\n" % (edge, i)
            self.cid2cell_type += "\t\treturn %d;\n\t}" % (snn.n_groups - 1)
            self.__cu_code = self.__cu_code.replace("$cid2cell_type$", self.cid2cell_type)
            self.__cu_code = self.__cu_code.replace("CELL_TYPE_INDEX[cid]", "cid2cell_type(cid)")
            self.__cu_code = self.__cu_code.replace("CELL_TYPE_INDEX[pre_cid]", "cid2cell_type(pre_cid)")
        else:
            self.__cu_code = self.__cu_code.replace("$cid2cell_type$", "")

    @property
    def cu_code(self):
        return self.__cu_code

    def __insert_local_variable(self):
        self.__cu_code = self.cu_code.replace("$allocate_var$", "\n\t\t".join(sorted(self.allocate_var)))

    def __add_update_func(self, calc_group: CellGroup, method, gm_swap):
        update_func = self.calc_group_temp
        # tokenize derivative code
        if method == "rk4":
            spike_receive, pre_update, update, spike_detection, spike_event = rk4_method(calc_group, gm_swap)
        else:
            spike_receive, pre_update, update, spike_detection, spike_event = euler_method(calc_group, gm_swap)     # euler method

        # insert post synaptic event template
        if len(calc_group.stdp_list) != 0 or calc_group.synapse is not None:
            update_func = update_func.replace("$spike_receive$", self.syn_event_tmp)

        # local variable allocation
        for var in calc_group.l_vars:
            if var.dtype == "int":
                self.allocate_var.add("%s %s = %d;" % (var.dtype, var.name, var.ini_val))
            if var.dtype == "float":
                self.allocate_var.add("%s %s = %f;" % (var.dtype, var.name, var.ini_val))
        for line in calc_group.allocation:
            self.allocate_var.add(line)

        # replace blocks
        update_func = update_func.replace("$pre_loop$", "\n\t\t\t".join(calc_group.pre_loop))
        update_func = update_func.replace("$spike_receive$", "\n\t\t\t".join(spike_receive))
        update_func = update_func.replace("$pre_update$", "\n\t\t\t\t".join(pre_update))
        update_func = update_func.replace("$update$", "\n\t\t\t\t".join(update))
        update_func = update_func.replace("$spike_detection$", " && ".join(spike_detection))
        update_func = update_func.replace("$spike_event$", "\n\t\t\t\t\t".join(spike_event))
        update_func = update_func.replace("$post_loop$", "\n\t\t\t".join(calc_group.post_loop))

        if len(calc_group.stdp_list) == 0:
            update_func = update_func.replace("$stdp_pre_post$", "")
            update_func = update_func.replace("$stdp_post_pre$", "")

        return update_func

    def __make_cu_code(self):
        self.__cu_code = self.__base.replace("$constant_memory$", self.constant_memory)
        self.__cu_code = self.__cu_code.replace("$args$", ", ".join(self.gpu_args))
        self.__cu_code = self.__cu_code.replace("$calc_group$", self.calc_group)

    def __replace_stdp(self, snn, gm_swap):
        stdp_pre_post = ''
        stdp_post_pre = ''
        w_update = ''
        for i, stdp in enumerate(snn.stdp_groups):
            if i > 0:   # i == 0 for None(no STDP synapse)
                stdp_pre_post += "\n" + "\t" * 7 + "if(" + " || ".join(["($NUM_CELL_TYPE$*pre_cti+post_cti == %d)" % idx for idx in snn.stdp_idx[i]]) + "){%s}" % stdp.pre_post
                stdp_post_pre += "\n" + "\t" * 6 + "if(" + " || ".join(["($NUM_CELL_TYPE$*post_cti+pre_cti == %d)" % idx for idx in snn.stdp_idx[i]]) + "){%s}" % stdp.post_pre
                w_update += "\n" + "\t" * 4 + "if(" + " || ".join(["($NUM_CELL_TYPE$*pre_cti+post_cti == %d)" % idx for idx in snn.stdp_idx[i]]) + "){%s}" % stdp.w_update
                # stdp_pre_post += "\n" + "\t" * 7 + "if(STDP_INDEX[$NUM_CELL_TYPE$ * pre_cti + post_cti] == %d){%s}" % (i, stdp.pre_post)
                # stdp_post_pre += "\n" + "\t" * 6 + "if(STDP_INDEX[$NUM_CELL_TYPE$ * post_cti + pre_cti] == %d){%s}" % (i, stdp.post_pre)
                # w_update += "\n" + "\t" * 4 + "if(STDP_INDEX[$NUM_CELL_TYPE$ * pre_cti + post_cti] == %d){%s}" % (i, stdp.w_update)

        if gm_swap:
            stdp_pre_post = stdp_swap(stdp_pre_post, list(self.s_var_name))
            stdp_post_pre = stdp_swap(stdp_post_pre, list(self.s_var_name))

        self.__cu_code = self.__cu_code.replace("$stdp_pre_post$", stdp_pre_post)
        self.__cu_code = self.__cu_code.replace("$stdp_post_pre$", stdp_post_pre)
        self.__cu_code = self.__cu_code.replace("$w_update$", w_update)

    def __replace_simulation_params(self, simulator):
        self.__cu_code = self.__cu_code.replace("$CELL_NUM$", str(simulator.snn.n_cells))
        self.__cu_code = self.__cu_code.replace("$NUM_CELL_TYPE$", str(simulator.snn.n_groups))
        self.__cu_code = self.__cu_code.replace("$NUM_CELL_TYPE2$", str(simulator.snn.n_groups ** 2))
        self.__cu_code = self.__cu_code.replace("$DT$", str(simulator.dt) + 'f')
        self.__cu_code = self.__cu_code.replace("$DT_DIV_2$", str(simulator.dt / 2) + 'f')
        self.__cu_code = self.__cu_code.replace("$DT_DIV_6$", str(simulator.dt / 6) + 'f')
        self.__cu_code = self.__cu_code.replace("$BLOCK_SIZE$", str(simulator.block_size))
        self.__cu_code = self.__cu_code.replace("$GRID_SIZE$", str(simulator.grid_size))
        self.__cu_code = self.__cu_code.replace("$MAX_SPIKE_IN_STEP$", str(simulator.max_spike_in_step))
        self.__cu_code = self.__cu_code.replace("$DELAY_STEP$", str(int(simulator.syn_delay / simulator.dt)))
        self.__cu_code = self.__cu_code.replace("$WIJ_UPDATE_INTERVAL$", str(simulator.wij_update_interval))
