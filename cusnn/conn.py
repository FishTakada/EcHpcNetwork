import numpy as np
from .cells import CellGroup

__all__ = ["Connect", "conn_all2all", "conn_all2number", "conn_number2all", "conn_all2all_uniform"]


class Connect:
    def __init__(self, seed=0):
        self.rs = np.random.RandomState(seed)

    def conn_all2all(self, pre: CellGroup, post: CellGroup, prob, w, no_self=False):
        wij = self.rs.rand(pre.n_cells, post.n_cells).astype(dtype=np.float32)
        wij[wij > prob] = 0
        wij[wij != 0] = w
        if no_self:
            wij *= ~np.eye(pre.n_cells, dtype=np.bool)  # remove self connection
        return wij

    def conn_all2all_uniform(self, pre: CellGroup, post: CellGroup, prob, w_min, w_max, no_self=False):
        wij = self.rs.rand(pre.n_cells, post.n_cells).astype(dtype=np.float32)
        wij[wij > prob] = 0
        wij[wij != 0] = 1
        wij *= self.rs.uniform(w_min, w_max, (pre.n_cells, post.n_cells)).astype(dtype=np.float32)
        if no_self:
            wij *= ~np.eye(pre.n_cells, dtype=np.bool)  # remove self connection
        return wij

    def conn_all2number(self, pre: CellGroup, post: CellGroup, syn_num, w, no_self=False):
        wij = np.zeros((pre.n_cells, post.n_cells), dtype=np.float32)
        for i in range(pre.n_cells):
            index = self.rs.randint(0, post.n_cells, syn_num)
            wij[i, index] = w
        if no_self:
            wij *= ~np.eye(pre.n_cells, dtype=np.bool)  # remove self connection
        return wij

    def conn_number2all(self, pre: CellGroup, post: CellGroup, syn_num, w, no_self=False):
        wij = np.zeros((pre.n_cells, post.n_cells), dtype=np.float32)
        for i in range(post.n_cells):
            index = self.rs.randint(0, pre.n_cells, syn_num)
            wij[index, i] = w
        if no_self:
            wij *= ~np.eye(pre.n_cells, dtype=np.bool)  # remove self connection
        return wij


conn = Connect()
conn_all2all = conn.conn_all2all
conn_all2number = conn.conn_all2number
conn_number2all = conn.conn_number2all
conn_all2all_uniform = conn.conn_all2all_uniform
