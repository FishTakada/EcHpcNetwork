import numpy as np
from .value_obj import Variable, LocalVariable, GlobalVariable, StateVariable
from typing import Optional, List, Union

__all__ = ["CellGroup", "Lif", "LifRefrac", "LifSpontaneous", "Izhikevich2007", "HH"]


class CodeBlock:
    def __init__(self):
        # code blocks
        self.allocation = set()
        self.pre_loop = set()
        self.spike_receive = set()
        self.pre_update = set()
        self.update = []
        self.spike_detection = set()
        self.spike_event = set()
        self.post_loop = set()


class CellGroup:
    def __init__(self, n_cells: int, **params):
        # parameter
        self.gid = None
        self.params = params
        self.n_cells = n_cells
        self.cell_type = 0

        # variables
        self.vars: List[Variable] = []
        self.s_vars: List[StateVariable] = []
        self.l_vars: List[LocalVariable] = []
        self.g_vars: List[Union[GlobalVariable, StateVariable]] = []
        self.var_names: List[str] = []

        # code blocks
        self.allocation = set()
        self.pre_loop = set()
        self.spike_receive = set()
        self.pre_update = set()
        self.update = []
        self.spike_detection = set()
        self.spike_event = set()
        self.post_loop = set()

        # synaptic property
        self.synapse = None
        self.synapse_list = []
        self.stdp_post = None
        self.stdp_list = []

        # add default variable
        self.add_var(LocalVariable(name="i_syn", dtype="float", init_val=0))
        self.add_var(LocalVariable(name="i_ext", dtype="float", init_val=0))
        self.add_var(LocalVariable(name="g_syn", dtype="float", init_val=0))
        self.spike_event.add("spike_log[cid] = d_step;")        # send spike signal

    def add_var(self, new_var: Variable):
        if new_var.name in self.var_names:
            return
        self.vars.append(new_var)
        self.var_names.append(new_var.name)
        if type(new_var) == StateVariable:
            self.s_vars.append(new_var)
            self.g_vars.append(new_var)
        elif type(new_var) == GlobalVariable:
            self.g_vars.append(new_var)
        else:
            self.l_vars.append(new_var)


class Izhikevich2007(CellGroup):
    def __init__(self, n_cells: int, **params):
        super().__init__(n_cells, **params)

        # parameters
        b, c = params.get("b", 0.2), params.get("c", -65.0)

        # define spike events
        self.spike_detection.add("v >= vth".replace("vth", str(params.get('vth', 30.0))))
        self.spike_event.add("v = c;".replace("c", str(c)))
        self.spike_event.add("u += d;".replace("d", str(params.get("d", 8.0))))

        # state variables
        self.v = StateVariable(name="v", dtype="float", derivative="(0.04 * v + 5) * v + 140 - u + 2.0 * i_syn + i_ext", init_val=c)
        self.u = StateVariable(name="u", dtype="float", derivative="a * (b * v - u)".
                               replace("a", str(params.get("a", 0.02))).replace("b", str(b)), init_val=b*c)
        self.add_var(self.v)
        self.add_var(self.u)


class Lif(CellGroup):
    def __init__(self, n_cells: int, **params):
        super().__init__(n_cells, **params)

        # define spike events
        self.spike_detection.add("v >= vth".replace("vth", str(params.get("vth", 1.))))
        self.spike_event.add("v = 0.0;")

        # state variable
        self.v = StateVariable(name="v", dtype="float", init_val=0, derivative="(-v/R + i_syn + i_ext) / C".
                               replace("R", str(params.get("r", 6.))).replace("C", str(params.get("c", 8.))))
        self.add_var(self.v)


class LifRefrac(Lif):
    def __init__(self, n_cells: int, **params):
        super().__init__(n_cells, **params)

        # define spike events
        self.spike_detection.add("refractory <= 0.0")
        self.spike_event.add("refractory = (float)($refrac$/$DT$);".replace("$refrac$", str(params.get("refrac", 0))))

        # refractory
        self.pre_update.add("if(refractory > 0)refractory -= 1.0f;")     # update in pre-update block
        self.refractory = StateVariable(name="refractory", dtype="float", init_val=0, derivative="0.0")
        self.add_var(self.refractory)


class LifSpontaneous(CellGroup):
    def __init__(self, n_cells: int, **params):
        super().__init__(n_cells, **params)

        # define spike events
        self.spike_detection.add("v >= vth".replace("vth", str(params.get("vth", 1.))))
        self.spike_event.add("v = 0.0;")

        # state variable
        self.v = StateVariable(name="v", dtype="float", init_val=0, derivative="(-v/R + i_syn + i_ext + i_sp) / C".
                               replace("R", str(params.get("r", 6.))).replace("C", str(params.get("c", 8.))).
                               replace("i_sp", str(params.get("i_spontaneous", 0.5))))
        self.add_var(self.v)


class HH(CellGroup):
    def __init__(self, n_cells: int, **params):
        super().__init__(n_cells, **params)
        self.c = params.get("c", 1)     # microF/cm2
        self.g_na = params.get("g_na", 120.)     # mS/cm2
        self.g_k = params.get("g_k", 36.)  # mS/cm2
        self.g_leak = params.get("g_leak", 0.3)  # mS/cm2
        self.e_na = params.get("e_na", 50.)  # mV
        self.e_k = params.get("e_k", -77.)  # mV
        self.e_leak = params.get("e_leak", -54.4)  # mV
        self.v_ini = params.get("v_ini", -70.0)  # mV
        self.v_th = params.get("v_th", 30.0)    # mV

        # calc initial state
        alpha, beta = 0.1 * (self.v_ini + 40) / (1-np.exp(-(self.v_ini+40)/10)), 4*np.exp(-(self.v_ini+65)/18)
        m_inf = alpha / (alpha + beta)
        alpha, beta = (0.01*(self.v_ini+55))/(1-np.exp(-(self.v_ini+55)/10)), 0.125*np.exp(-(self.v_ini+65)/80)
        n_inf = alpha / (alpha + beta)
        alpha, beta = 0.07*np.exp(-(self.v_ini+65)/20), 1/(1+np.exp(-(self.v_ini+35)/10))
        h_inf = alpha / (alpha + beta)

        # state variables
        self.v = StateVariable(name="v", dtype="float", init_val=self.v_ini, derivative="(g_na*m*m*m*h*(e_na-v) + g_k*n*n*n*n*(e_k-v) + g_leak*(e_leak-v) + i_syn + i_ext)/C")
        self.m = StateVariable(name="m", dtype="float", init_val=m_inf, derivative="(0.1*(v+40)/(1-expf(-(v+40)/10)))*(1-m) - (4*expf(-(v+65)/18))*m")
        self.n = StateVariable(name="n", dtype="float", init_val=n_inf, derivative="(0.01*(v+55)/(1-expf(-(v+55)/10)))*(1-n) - (0.125*expf(-(v+65)/80))*n")
        self.h = StateVariable(name="h", dtype="float", init_val=h_inf, derivative="(0.07*expf(-(v+65)/20))*(1-h) - (1/(1+expf(-(v+35)/10)))*h")
        self.v.derivative = self.v.derivative.replace("g_na", str(self.g_na)).replace("g_k", str(self.g_k)).replace("g_leak", str(self.g_leak)). \
            replace("e_na", str(self.e_na)).replace("e_k", str(self.e_k)).replace("e_leak", str(self.e_leak)).replace("C", str(self.c))
        self.add_var(self.v)
        self.add_var(self.m)
        self.add_var(self.n)
        self.add_var(self.h)

        # v_past for spike detection
        self.add_var(LocalVariable(name="v_past", dtype="float", init_val=0))
        self.pre_update.add("v_past = v;")
        self.spike_detection.add("v_past < $v_th$ && v >= $v_th$".replace("$v_th$", str(self.v_th)).replace("$v_th$", str(self.v_th)))

