import numpy as np
import cusnn as cs
from cusnn.value_obj import StateVariable, LocalVariable, GlobalVariable

pi2 = np.pi * 2


class EcHpc(cs.Module):
    def __init__(self, params: dict):
        super().__init__()

        # parameters and definitions
        self.n_dir = 4
        self.n_action = 16
        self.nx, self.ny = 30, 26
        self.sl_scale = 1.0
        self.sl_ny, self.sl_nx = int(self.ny * self.sl_scale), int(self.nx * self.sl_scale)
        self.sl_size = self.sl_nx * self.sl_ny
        self.head_dir = np.array([(pi2 / self.n_dir * i) for i in range(self.n_dir)])
        self.action_dir = np.array([(pi2 / self.n_action * i) for i in range(self.n_action)])
        self.conn = cs.Connect(seed=params.get("conn_seed", 0))
        self.torus = np.array(
            [[0, 0], [self.ny, -self.nx / 2], [self.ny, self.nx / 2], [0, -self.nx], [0, self.nx], [-self.ny, -self.nx / 2], [-self.ny, self.nx / 2]],
            dtype=np.float32)

        # make cell groups
        # MEC
        self.grid_1: cs.LifRefrac = self._add_group(cs.LifRefrac(self.nx * self.ny, r=6., c=8., refrac=5))
        self.grid_2: cs.LifRefrac = self._add_group(cs.LifRefrac(self.nx * self.ny, r=6., c=8., refrac=5))
        self.inhibi_1: cs.LifRefrac = self._add_group(cs.LifRefrac(1, r=6., c=8., refrac=2))
        self.inhibi_2: cs.LifRefrac = self._add_group(cs.LifRefrac(1, r=6., c=8., refrac=2))
        self.slide_1: cs.LifRefrac = self._add_group(cs.LifRefrac(self.sl_nx * self.sl_ny * self.n_dir, r=6., c=8., refrac=5))
        self.slide_2: cs.LifRefrac = self._add_group(cs.LifRefrac(self.sl_nx * self.sl_ny * self.n_dir, r=6., c=8., refrac=5))
        self.padding: cs.LifRefrac = self._add_group(cs.LifRefrac(2, r=6., c=8., refrac=5))
        self.head_1: cs.LifRefrac = self._add_group(cs.LifRefrac(self.n_dir, r=6., c=8., refrac=5))
        self.head_2: cs.LifRefrac = self._add_group(cs.LifRefrac(self.n_dir, r=6., c=8., refrac=5))
        # HPC
        self.pyramidal: cs.LifRefrac = self._add_group(cs.LifRefrac(3600, r=6., c=8., refrac=5, tau_da=200.))
        self.basket: cs.LifRefrac = self._add_group(cs.LifRefrac(400, r=6., c=8., refrac=5))
        # AC
        self.action_neuron: cs.LifRefrac = self._add_group(cs.LifRefrac(self.n_action, r=6., c=8., refrac=5))
        self.action_inhibi: cs.LifRefrac = self._add_group(cs.LifRefrac(self.n_action, r=6., c=8., refrac=5))

        # make network
        self._make_network()

    def _make_connection(self):
        # synapse model
        syn = cs.DoubleExpSyn(tau_ampa1=6., tau_ampa2=1., tau_gaba1=6., tau_gaba2=1., e_ampa=2., e_gaba=-1.)
        syn_in = cs.DoubleExpSyn(tau_ampa1=1.0, tau_ampa2=0.5, tau_gaba1=6., tau_gaba2=1., e_ampa=2., e_gaba=-1.)

        # MEC
        grid_x, grid_y = self._calc_grid_xy(self.nx, self.ny)
        slide_x, slide_y = self._calc_grid_xy(self.sl_nx, self.sl_ny, self.sl_scale)
        w_g2sl = np.zeros((self.grid_1.n_cells, self.slide_1.n_cells), dtype=np.float32)
        w_sl2g = np.zeros((self.slide_1.n_cells, self.grid_1.n_cells), dtype=np.float32)
        w_h2sl = np.zeros((self.head_1.n_cells, self.slide_1.n_cells), dtype=np.float32)
        for i in range(self.n_dir):
            w_g2sl[:, self.sl_size * i:self.sl_size * (i + 1)] = self.conn_grid2slide(self.grid_1.n_cells, self.sl_size, grid_x, grid_y, slide_x,
                                                                                      slide_y, w=0.0025, radius=3.)
            w_sl2g[self.sl_size * i:self.sl_size * (i + 1), :] = self.conn_slide2grid(self.sl_size, self.grid_1.n_cells, grid_x, grid_y,
                                                                                      slide_x, slide_y, self.head_dir[i], 4, w=0.01, radius=8.)
            w_h2sl[i, self.sl_size * i:self.sl_size * (i + 1)] = 0.25
        # grid 1
        self._add_connection(self.grid_1, self.grid_1, self.conn_grid2grid(self.grid_1, grid_x, grid_y, w=1.0, sigma=3), synapse=syn,
                             stdp=None)
        self._add_connection(self.grid_1, self.inhibi_1, cs.conn_all2all(self.grid_1, self.inhibi_1, prob=1., w=.05), synapse=syn_in, stdp=None)
        self._add_connection(self.inhibi_1, self.grid_1, cs.conn_all2all(self.inhibi_1, self.grid_1, prob=1., w=-0.72), synapse=syn, stdp=None)
        self._add_connection(self.head_1, self.slide_1, w_h2sl, synapse=syn, stdp=None)
        self._add_connection(self.grid_1, self.slide_1, w_g2sl, synapse=syn, stdp=None)
        self._add_connection(self.slide_1, self.grid_1, w_sl2g, synapse=syn, stdp=None)
        # grid 2
        self._add_connection(self.grid_2, self.grid_2, self.conn_grid2grid(self.grid_2, grid_x, grid_y, w=1.0, sigma=3), synapse=syn,
                             stdp=None)
        self._add_connection(self.grid_2, self.inhibi_2, cs.conn_all2all(self.grid_2, self.inhibi_2, prob=1., w=.05), synapse=syn_in, stdp=None)
        self._add_connection(self.inhibi_2, self.grid_2, cs.conn_all2all(self.inhibi_2, self.grid_2, prob=1., w=-0.72), synapse=syn, stdp=None)
        self._add_connection(self.head_2, self.slide_2, w_h2sl, synapse=syn, stdp=None)
        self._add_connection(self.grid_2, self.slide_2, w_g2sl, synapse=syn, stdp=None)
        self._add_connection(self.slide_2, self.grid_2, w_sl2g, synapse=syn, stdp=None)

        # HPC
        self._add_connection(self.grid_1, self.pyramidal, self.conn.conn_number2all(self.grid_1, self.pyramidal, syn_num=1, w=0.16), syn, None)
        self._add_connection(self.grid_2, self.pyramidal, self.conn.conn_number2all(self.grid_2, self.pyramidal, syn_num=1, w=0.16), syn, None)
        self._add_connection(self.pyramidal, self.pyramidal,
                             self.conn.conn_all2all_uniform(self.pyramidal, self.pyramidal, prob=0.16, w_min=0.0, w_max=0.001, no_self=True), syn,
                             DaStdpBiased(a_plus=0.12, a_minus=-0.3, tau_trace=5000., stdp_max=1.0, tau_plus=50.0, tau_minus=50.0,
                                          tau_da=200.0, bias=0.0))
        self._add_connection(self.pyramidal, self.basket, self.conn.conn_all2all(self.pyramidal, self.basket, prob=0.05, w=0.375), syn, None)
        self._add_connection(self.basket, self.pyramidal, self.conn.conn_all2all(self.basket, self.pyramidal, prob=0.05, w=-0.16), syn, None)
        self._add_connection(self.basket, self.basket, self.conn.conn_all2all(self.basket, self.basket, prob=0.01, w=-0.4, no_self=True), syn, None)

        # Action selection network
        self._add_connection(self.pyramidal, self.action_neuron,
                             self.conn.conn_all2all_uniform(self.pyramidal, self.action_neuron, prob=1., w_min=0.0, w_max=0.05), syn,
                             DaStdpBiased(a_plus=0.03, a_minus=0.03, tau_trace=1000., stdp_max=0.05, tau_plus=50.0, tau_minus=50.0,
                                          tau_da=200.0, bias=0.00015))
        self._add_connection(self.action_neuron, self.action_neuron, self.conn_action2action(w=1.), syn, None)
        self._add_connection(self.action_neuron, self.action_inhibi, np.eye(self.n_action, dtype=np.float32) * 0.5, syn, None)
        self._add_connection(self.action_inhibi, self.action_neuron, self.conn_inhibi2action(w=-1.5), syn, None)

    # ------------------------------- functions for making network ------------------------------- #
    @staticmethod
    def _calc_grid_xy(nx, ny, scale=1.):
        x, y = np.zeros(nx * ny, dtype=np.float32), np.zeros(nx * ny, dtype=np.float32)
        for i in range(ny):
            for j in range(nx):
                y[j * ny + i] = i + .5
                if i % 2 == 0:
                    x[j * ny + i] = j + .5
                else:
                    x[j * ny + i] = j + 1.
        return x * (1 / scale), y * (1 / scale)

    def conn_grid2grid(self, grid, grid_x, grid_y, w, sigma=3):
        def gaussian1d(x, sig):
            return np.exp(-np.power(x, 2.) / (2 * np.power(sig, 2.))) / (np.sqrt(pi2) * sig)

        # calc weight
        dist = np.zeros((7, grid.n_cells, grid.n_cells), dtype=np.float32)
        for i in range(7):
            dist[i] = 2 * np.sqrt(np.power(np.subtract.outer(grid_x, grid_x) + self.torus[i, 1], 2) +
                                  np.power(np.subtract.outer(grid_y, grid_y) + self.torus[i, 0], 2))
        dist_min = np.min(dist, axis=0)
        wij = gaussian1d(dist_min, sigma) * w
        wij *= ~np.eye(grid.n_cells, dtype=np.bool)  # remove self connection
        return wij

    def conn_grid2slide(self, grid_n_cells, slide_n_cells, grid_x, grid_y, slide_x, slide_y, w, radius=3.):
        wij = np.zeros((grid_n_cells, slide_n_cells), dtype=np.float32)
        for i in range(7):
            dist = np.sqrt(np.power(self.torus[i, 1] - np.subtract.outer(grid_x, slide_x), 2) +
                           np.power(self.torus[i, 0] - np.subtract.outer(grid_y, slide_y), 2))
            dist[dist > radius] = 0
            wij += dist
        wij[wij != 0] = w
        return wij

    def conn_slide2grid(self, slide_n_cells, grid_n_cells, grid_x, grid_y, slide_x, slide_y, p_dir, s_dir, w, radius=6.):
        dist = np.zeros((7, slide_n_cells, grid_n_cells), dtype=np.float32)
        for i in range(7):
            x = self.torus[i, 1] - np.subtract.outer(slide_x, grid_x) - s_dir * np.cos(p_dir)
            y = self.torus[i, 0] - np.subtract.outer(slide_y, grid_y) - s_dir * np.sin(p_dir)
            dist[i] = np.sqrt(np.power(x, 2) + np.power(y, 2))
        dist = np.min(dist, axis=0)
        mask = dist > radius
        wij = w * (1 + np.cos(np.pi * dist / radius)) * 0.5
        wij[mask] = 0.0
        return wij

    def conn_inhibi2action(self, w, sigma=2.4):
        dir_diff = np.abs(np.subtract.outer(self.action_dir, self.action_dir))
        dir_diff_mod = np.abs(np.subtract.outer(self.action_dir, self.action_dir + pi2))
        dir_diff[dir_diff > dir_diff_mod] = dir_diff_mod[dir_diff > dir_diff_mod]
        dir_diff_mod = np.abs(np.subtract.outer(self.action_dir, self.action_dir - pi2))
        dir_diff[dir_diff > dir_diff_mod] = dir_diff_mod[dir_diff > dir_diff_mod]
        wij = (w / np.sqrt(pi2 * sigma) * (1 - np.exp(-np.power(dir_diff, 2) / np.power(sigma, 2)))).astype(np.float32)
        wij *= ~np.eye(self.n_action, dtype=np.bool)
        return wij

    def conn_action2action(self, w, sigma=2.0):
        dir_diff = np.abs(np.subtract.outer(self.action_dir, self.action_dir))
        dir_diff_mod = np.abs(np.subtract.outer(self.action_dir, self.action_dir + pi2))
        dir_diff[dir_diff > dir_diff_mod] = dir_diff_mod[dir_diff > dir_diff_mod]
        dir_diff_mod = np.abs(np.subtract.outer(self.action_dir, self.action_dir - pi2))
        dir_diff[dir_diff > dir_diff_mod] = dir_diff_mod[dir_diff > dir_diff_mod]
        wij = (w / np.sqrt(pi2 * sigma) * np.exp(-np.power(dir_diff, 2) / np.power(sigma, 2))).astype(np.float32)
        wij *= ~np.eye(self.n_action, dtype=np.bool)
        return wij


class DaStdpBiased(cs.STDP):
    def __init__(self, **params):
        super().__init__(**params)
        self.a_plus = params.get("a_plus", 1.0)
        self.a_minus = params.get("a_minus", -1.0)
        self.tau_plus = params.get("tau_plus", 15.0)
        self.tau_minus = params.get("tau_minus", 30.0)
        self.stdp_min = params.get("stdp_min", 0.000001)
        self.stdp_max = params.get("stdp_max", 0.2)
        self.tau_trace = params.get("tau_trace", 1000.)
        self.__tau_da = params.get("tau_da", 200.)
        self.bias = params.get("bias", 0.002)
        self.mu = params.get("mu", 1.0)

        self.pre_post = """
                                atomicAdd(&trace[wij_pre_post], $A_MINUS$ * expf(-last_spike[cid] / $TAU_MINUS$));"""

        self.post_pre = """
                            wij_post_pre = WIJ_INDEX[$NUM_CELL_TYPE$ * post_cti + pre_cti] + post_iid * CELL_TYPE_WIDTH[pre_cti] + (pre_cid - CELL_TYPE_EDGE[pre_cti]);
                            atomicAdd(&trace[wij_post_pre], $A_PLUS$ * expf(-last_spike[cid]/$TAU_PLUS$));"""

        self.w_update = """
                    int pre_ctw = CELL_TYPE_WIDTH[pre_cti];
                    float f_ach_local = f_ach[cid];
                    for(short int pre_iid=0; pre_iid<pre_ctw; pre_iid++){
                        wij_pre_post = WIJ_INDEX[$NUM_CELL_TYPE$ * pre_cti + post_cti] + pre_iid * post_ctw + post_iid;
                        syn = wij[wij_pre_post];
                        if(syn > 0.0f){
                            syn += $WIJ_UPDATE_INTERVAL$ * ($DT$ * $DELAY_STEP$ * 0.1f) * f_ach_local * $MU$ * (da[cid] - $BIAS$) * trace[wij_pre_post];
                            if(syn < $STDP_MIN$ * f_ach_local){
                                syn = $STDP_MIN$;
                            }else if(syn > $STDP_MAX$ * f_ach_local){
                                syn = $STDP_MAX$;
                            }
                            wij[wij_pre_post] = syn;
                        }
                        trace[wij_pre_post] -= $WIJ_UPDATE_INTERVAL$ * $DELAY_STEP$ * $DT$ / $TAU_TRACE$ * trace[wij_pre_post];
                    }"""

        # dynamic variable
        self.da = StateVariable(name="da", dtype="float", init_val=0, derivative="-da / $tau_da$".replace("$tau_da$", str(self.__tau_da) + 'f'))
        self.last_spike = StateVariable(name="last_spike", dtype="float", init_val=100000, derivative="1.0")

        # synapse variable
        self.trace = GlobalVariable(name="trace", dtype="float", init_val=0, is_syn=True)
        self.f_ach = GlobalVariable(name="f_ach", dtype="float", init_val=1.0)

        self.pre_post = self.pre_post.replace("$A_MINUS$", str(self.a_minus) + 'f')
        self.pre_post = self.pre_post.replace("$TAU_MINUS$", str(self.tau_minus) + 'f')
        self.post_pre = self.post_pre.replace("$A_PLUS$", str(self.a_plus) + 'f')
        self.post_pre = self.post_pre.replace("$TAU_PLUS$", str(self.tau_plus) + 'f')
        self.w_update = self.w_update.replace("$STDP_MAX$", str(self.stdp_max) + 'f')
        self.w_update = self.w_update.replace("$STDP_MIN$", str(self.stdp_min) + 'f')
        self.w_update = self.w_update.replace("$TAU_TRACE$", str(self.tau_trace) + 'f')
        self.w_update = self.w_update.replace("$BIAS$", str(self.bias) + 'f')
        self.w_update = self.w_update.replace("$MU$", str(self.mu) + 'f')

    def insert(self, pre: cs.CellGroup, post: cs.CellGroup):
        # insert mechanism to the pre-cell-group
        pre.add_var(LocalVariable(name="wij_pre_post", dtype="int", init_val=0))
        pre.add_var(LocalVariable(name="wij_post_pre", dtype="int", init_val=0))
        pre.add_var(self.last_spike)
        pre.pre_update.add("if(last_spike[cid] > 100000.0)last_spike[cid] = 100000.0;")
        pre.spike_event.add("last_spike[cid] = 0.0f;")

        # insert mechanism to the post-cell-group
        post.add_var(self.trace)
        post.add_var(self.da)
        post.add_var(self.last_spike)
        post.add_var(self.f_ach)
        post.pre_update.add("if(last_spike[cid] > 100000.0)last_spike[cid] = 100000.0;")
        post.spike_event.add("last_spike[cid] = 0.0f;")


