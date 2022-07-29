from abc import ABCMeta, abstractmethod
from .value_obj import StateVariable, LocalVariable, GlobalVariable
from .cells import CellGroup

__all__ = ["STDP", "AsymmetricSTDP", "DaStdp"]


class STDP(metaclass=ABCMeta):
    def __init__(self, **params):
        self.params = params
        self.w_update = ""
        self.pre_post = ""
        self.post_pre = ""
        self.syn_vars = []

    # this function should be overwrite
    @abstractmethod
    def insert(self, pre: CellGroup, post: CellGroup):
        pass


class AsymmetricSTDP(STDP):
    def __init__(self, **params):
        super().__init__(**params)

        # parameters
        self.a_plus = params.get("a_plus", 1.0)
        self.a_minus = params.get("a_minus", -1.0)
        self.tau_plus = params.get("tau_plus", 15.0)
        self.tau_minus = params.get("tau_minus", 30.0)
        self.stdp_min = params.get("stdp_min", 0.000001)
        self.stdp_max = params.get("stdp_max", 0.2)

        # define stdp event
        self.pre_post = """
                                delta_w = $A_MINUS$ * expf(-last_spike[cid] / $TAU_MINUS$);
                                if((wij[wij_pre_post] > 0.0) && (wij[wij_pre_post] + delta_w > $STDP_MIN$)) atomicAdd(&wij[wij_pre_post], delta_w);"""

        self.post_pre = """
                            wij_post_pre = WIJ_INDEX[$NUM_CELL_TYPE$ * post_cti + pre_cti] + post_iid * CELL_TYPE_WIDTH[pre_cti] + (pre_cid - CELL_TYPE_EDGE[pre_cti]);
                            delta_w = $A_PLUS$ * expf(-last_spike[cid] / $TAU_PLUS$);
                            if((wij[wij_post_pre] > 0.0) && (wij[wij_post_pre] + delta_w < $STDP_MAX$)) atomicAdd(&wij[wij_post_pre], delta_w);"""
        self.pre_post = self.pre_post.replace("$A_MINUS$", str(self.a_minus) + 'f')
        self.pre_post = self.pre_post.replace("$TAU_MINUS$", str(self.tau_minus) + 'f')
        self.pre_post = self.pre_post.replace("$STDP_MIN$", str(self.stdp_min) + 'f')
        self.post_pre = self.post_pre.replace("$A_PLUS$", str(self.a_plus) + 'f')
        self.post_pre = self.post_pre.replace("$TAU_PLUS$", str(self.tau_plus) + 'f')
        self.post_pre = self.post_pre.replace("$STDP_MAX$", str(self.stdp_max) + 'f')

        # dynamic variable
        self.last_spike = StateVariable(name="last_spike", dtype="float", init_val=100000, derivative="1.0")

    def insert(self, pre: CellGroup, post: CellGroup):
        # insert mechanism to the pre-cell-group
        pre.add_var(LocalVariable(name="delta_w", dtype="float", init_val=0))
        pre.add_var(LocalVariable(name="wij_pre_post", dtype="int", init_val=0))
        pre.add_var(LocalVariable(name="wij_post_pre", dtype="int", init_val=0))
        pre.add_var(self.last_spike)
        pre.pre_update.add("if(last_spike > 100000.0)last_spike = 100000.0;")
        pre.spike_event.add("last_spike = 0.0;")

        # insert mechanism to the post-cell-group
        post.add_var(self.last_spike)
        post.pre_update.add("if(last_spike > 100000.0)last_spike = 100000.0;")
        post.spike_event.add("last_spike = 0.0f;")


# Izhikevich (2007)
class DaStdp(STDP):
    def __init__(self, **params):
        super().__init__(**params)

        # parameters
        self.a_plus = params.get("a_plus", 1.0)
        self.a_minus = params.get("a_minus", -1.0)
        self.tau_plus = params.get("tau_plus", 15.0)
        self.tau_minus = params.get("tau_minus", 30.0)
        self.stdp_min = params.get("stdp_min", 0.000001)
        self.stdp_max = params.get("stdp_max", 0.2)
        self._tau_da = params.get("tau_da", 200.0)
        self.tau_trace = params.get("tau_trace", 1000.)
        self.bias = params.get("bias", 0.002)

        # define stdp event
        self.pre_post = """
                                atomicAdd(&trace[wij_pre_post], $A_MINUS$ * expf(-last_spike[cid] / $TAU_MINUS$));"""

        self.post_pre = """
                            wij_post_pre = WIJ_INDEX[$NUM_CELL_TYPE$ * post_cti + pre_cti] + post_iid * CELL_TYPE_WIDTH[pre_cti] + (pre_cid - CELL_TYPE_EDGE[pre_cti]);
                            atomicAdd(&trace[wij_post_pre], $A_PLUS$ * expf(-last_spike[cid]/$TAU_PLUS$));"""

        self.w_update = """
                    int pre_ctw = CELL_TYPE_WIDTH[pre_cti];
                    for(short int pre_iid=0; pre_iid<pre_ctw; pre_iid++){
                        wij_pre_post = WIJ_INDEX[$NUM_CELL_TYPE$ * pre_cti + post_cti] + pre_iid * post_ctw + post_iid;
                        syn = wij[wij_pre_post];
                        if(syn > 0.0f){
                            syn += $WIJ_UPDATE_INTERVAL$ * ($DT$ * $DELAY_STEP$ * 0.1f) * ($BIAS$ + da[cid]) * trace[wij_pre_post];
                            if(syn < $STDP_MIN$){
                                syn = $STDP_MIN$;
                            }else if(syn > $STDP_MAX$){
                                syn = $STDP_MAX$;
                            }
                            wij[wij_pre_post] = syn;
                        }
                        trace[wij_pre_post] -= $WIJ_UPDATE_INTERVAL$ * $DELAY_STEP$ * $DT$ / $TAU_TRACE$ * trace[wij_pre_post];
                    }"""
        self.pre_post = self.pre_post.replace("$A_MINUS$", str(self.a_minus) + 'f')
        self.pre_post = self.pre_post.replace("$TAU_MINUS$", str(self.tau_minus) + 'f')
        self.post_pre = self.post_pre.replace("$A_PLUS$", str(self.a_plus) + 'f')
        self.post_pre = self.post_pre.replace("$TAU_PLUS$", str(self.tau_plus) + 'f')
        self.w_update = self.w_update.replace("$STDP_MAX$", str(self.stdp_max) + 'f')
        self.w_update = self.w_update.replace("$STDP_MIN$", str(self.stdp_min) + 'f')
        self.w_update = self.w_update.replace("$TAU_TRACE$", str(self.tau_trace) + 'f')
        self.w_update = self.w_update.replace("$BIAS$", str(self.bias) + 'f')

        # dynamic variable
        self.da = StateVariable(name="da", dtype="float", init_val=0, derivative="-da / $tau_da$".replace("$tau_da$", str(self._tau_da) + 'f'))
        self.last_spike = StateVariable(name="last_spike", dtype="float", init_val=100000, derivative="1.0")

        # synapse variable
        self.trace = GlobalVariable(name="trace", dtype="float", init_val=0, is_syn=True)

    def insert(self, pre: CellGroup, post: CellGroup):
        # insert mechanism to the pre-cell-group
        pre.add_var(LocalVariable(name="wij_pre_post", dtype="int", init_val=0))
        pre.add_var(LocalVariable(name="wij_post_pre", dtype="int", init_val=0))
        pre.add_var(self.last_spike)
        pre.add_var(self.trace)
        pre.pre_update.add("if(last_spike > 100000.0)last_spike = 100000.0;")
        pre.spike_event.add("last_spike = 0.0f;")

        # insert mechanism to the post-cell-group
        post.add_var(self.da)
        post.add_var(self.last_spike)
        post.pre_update.add("if(last_spike > 100000.0)last_spike = 100000.0;")
        post.spike_event.add("last_spike = 0.0f;")
