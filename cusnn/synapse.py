from abc import ABCMeta, abstractmethod
from .value_obj import StateVariable, LocalVariable, GlobalVariable
from .cells import CellGroup

__all__ = ["Synapse", "CurrentSyn", "DoubleExpSyn"]


class Synapse(metaclass=ABCMeta):
    def __init__(self, **params):
        self.params = params

    # this function should be overwrite
    @abstractmethod
    def insert(self, pre: CellGroup, post: CellGroup):
        pass


class CurrentSyn(Synapse):
    def __init__(self, **params):
        super().__init__(**params)

    def insert(self, pre: CellGroup, post: CellGroup):
        # insert mechanism to the post-cell-group
        post.spike_receive.add("i_syn += g_syn;")
        post.add_var(LocalVariable(name="wij_pre_post", dtype="int", init_val=0))


class DoubleExpSyn(Synapse):
    def __init__(self, **params):
        super().__init__(**params)

        # synaptic receptor parameter
        self._tau_ampa1 = params.get("tau_ampa1", 6.)
        self._tau_ampa2 = params.get("tau_ampa2", 1.)
        self._tau_gaba1 = params.get("tau_gaba1", 6.)
        self._tau_gaba2 = params.get("tau_gaba2", 1.)
        self._e_ampa = params.get("e_ampa", 2.)
        self._e_gaba = params.get("e_gaba", -1.)

        # define post synaptic current
        self.spike_receive = """
        if(g_syn > 0.0f){
            a_ampa += g_syn;
            b_ampa += g_syn;
        }else{
            a_gaba -= g_syn;
            b_gaba -= g_syn;
        }"""

        self.pre_update = """
                i_syn = (b_ampa - a_ampa) * ($E_AMPA$ - v);
                i_syn += (b_gaba - a_gaba) * ($E_GABA$ - v);""".\
            replace("$E_AMPA$", str(self._e_ampa) + 'f').replace("$E_GABA$", str(self._e_gaba) + 'f')

        # dynamic variable
        self.a_ampa = StateVariable(name="a_ampa", dtype="float", init_val=0, derivative="-a_ampa * $ONE_DIV_TAU_AMPA2$".
                                    replace("$ONE_DIV_TAU_AMPA2$", str(1 / self._tau_ampa2) + 'f'))
        self.b_ampa = StateVariable(name="b_ampa", dtype="float", init_val=0, derivative="-b_ampa * $ONE_DIV_TAU_AMPA1$".
                                    replace("$ONE_DIV_TAU_AMPA1$", str(1 / self._tau_ampa1) + 'f'))
        self.a_gaba = StateVariable(name="a_gaba", dtype="float", init_val=0, derivative="-a_gaba * $ONE_DIV_TAU_GABA2$".
                                    replace("$ONE_DIV_TAU_GABA2$", str(1 / self._tau_gaba2) + 'f'))
        self.b_gaba = StateVariable(name="b_gaba", dtype="float", init_val=0, derivative="-b_gaba * $ONE_DIV_TAU_GABA1$".
                                    replace("$ONE_DIV_TAU_GABA1$", str(1 / self._tau_gaba1) + 'f'))

    def insert(self, pre: CellGroup, post: CellGroup):
        # insert mechanism to the post-cell-group
        post.spike_receive.add(self.spike_receive)
        post.pre_update.add(self.pre_update)
        post.add_var(LocalVariable(name="wij_pre_post", dtype="int", init_val=0))
        post.add_var(self.a_ampa)
        post.add_var(self.b_ampa)
        post.add_var(self.a_gaba)
        post.add_var(self.b_gaba)