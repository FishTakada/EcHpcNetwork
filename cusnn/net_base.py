import numpy as np
from abc import ABCMeta, abstractmethod
from .cells import CellGroup
from .value_obj import Weight
from .synapse import Synapse
from .stdp import STDP
from typing import List, Union, Optional

__all__ = ["Module"]


class Module(metaclass=ABCMeta):
    def __init__(self):
        self.__cell_groups = []
        self.__calc_groups = None
        self.__stdp_groups = [None]
        self.stdp_idx = [[]]
        self.__calc_group_n_cells = None
        self.__n_syns = 0
        self.wij = None
        self.wij_index = None
        self.__syn_list = []
        self.__stdp_list = []
        self.has_update_weight = False

    @property
    def cell_groups(self) -> List[CellGroup]:
        return self.__cell_groups

    @property
    def n_groups(self):
        return len(self.__cell_groups)

    @property
    def n_cells(self):
        return sum(groups.n_cells for groups in self.__cell_groups)

    @property
    def n_syns(self):
        return self.__n_syns

    @property
    def calc_groups(self):
        return self.__calc_groups

    @property
    def stdp_groups(self) -> List[Optional[STDP]]:
        return self.__stdp_groups

    @property
    def calc_group_n_cells(self):
        return self.__calc_group_n_cells

    @property
    def syn_list(self) -> List[tuple]:
        return self.__syn_list

    @property
    def stdp_list(self):
        return self.__stdp_list

    @abstractmethod
    def _make_connection(self):
        # this function must be override
        pass

    def _make_network(self):
        self._make_connection()
        self.__sort_network()
        self.classify_stdp()
        self.__make_weight()

    def __sort_network(self):
        # make calc groups
        self.__calc_groups = [self.cell_groups[0]]
        self.__calc_group_n_cells = [self.cell_groups[0].n_cells]
        cell2calc_index = [0]
        cnt = 1
        for i in range(1, self.n_groups):
            for j in range(len(self.__calc_groups)):
                # if same class and same parameter
                if self.compare_obj(self.cell_groups[i], self.__calc_groups[j]) and \
                        self.compare_obj(self.cell_groups[i].synapse, self.__calc_groups[j].synapse) and \
                        self.compare_obj(self.cell_groups[i].stdp_post, self.__calc_groups[j].stdp_post) and \
                        ((len(self.cell_groups[i].stdp_list) == 0) is (len(self.__calc_groups[j].stdp_list) == 0)):
                    cell2calc_index.append(j)
                    self.__calc_group_n_cells[j] += self.cell_groups[i].n_cells
                    break
            else:
                self.__calc_groups.append(self.cell_groups[i])
                self.__calc_group_n_cells.append(self.cell_groups[i].n_cells)
                cell2calc_index.append(cnt)
                cnt += 1
        self.__cell_groups = [self.__cell_groups[i] for i in sorted(range(self.n_groups), key=lambda x: cell2calc_index[x])]
        for i, cell_group in enumerate(self.__cell_groups):
            cell_group.cell_type = i

        # assign gid to cell_groups
        cnt = 0
        for cell_group in self.__cell_groups:
            cell_group.gid = np.arange(cnt, cnt + cell_group.n_cells, dtype=np.int32)
            cnt += cell_group.n_cells

    def classify_synapse(self):
        # stdp_list = [(pre: CellGroup, post: CellGroup, stdp:STDP), ..., ()]
        cnt = 1
        for pre, post, stdp in self.stdp_list:
            for k in range(len(self.__stdp_groups)):
                if self.compare_obj(stdp, self.__stdp_groups[k]):
                    self.stdp_idx[k].append(self.n_groups*pre.cell_type+post.cell_type)
                    break
            else:
                self.__stdp_groups.append(stdp)
                self.stdp_idx.append([self.n_groups*pre.cell_type+post.cell_type])
                cnt += 1

    def classify_stdp(self):
        # stdp_list = [(pre: CellGroup, post: CellGroup, stdp:STDP), ..., ()]
        cnt = 1
        for pre, post, stdp in self.stdp_list:
            for k in range(len(self.__stdp_groups)):
                if self.compare_obj(stdp, self.__stdp_groups[k]):
                    self.stdp_idx[k].append(self.n_groups*pre.cell_type+post.cell_type)
                    break
            else:
                self.__stdp_groups.append(stdp)
                self.stdp_idx.append([self.n_groups*pre.cell_type+post.cell_type])
                cnt += 1

    def __make_weight(self):
        # make 1d weight array
        wij_flat = np.array([], dtype=np.float32)
        wij_index = np.ones((self.n_groups, self.n_groups), dtype=np.int32) * -1
        for pre, post, weight in self.syn_list:
            wij_index[pre.cell_type, post.cell_type] = self.__n_syns
            self.__n_syns += weight.size
            # wij_flat = np.concatenate((wij_flat, weight.flatten()))
        if len(self.syn_list) != 0:
            wij_flat = np.concatenate([w for _, _, w in self.syn_list])
        self.wij = Weight(wij_flat, wij_index)

    def _add_group(self, cell_group: CellGroup):
        self.__cell_groups.append(cell_group)
        return cell_group

    def _add_connection(self, pre: CellGroup, post: CellGroup, weight: np.ndarray, synapse: Synapse, stdp: Optional[STDP]):
        # add post-synaptic process to the post cell group
        if post.synapse is None:
            post.synapse = synapse
            synapse.insert(pre, post)
        elif not self.compare_obj(synapse, post.synapse):
            print("different synapse type detected on one cell group!")
            exit()
        self.__syn_list.append((pre, post, weight.flatten()))

        # add post-synaptic stdp process to the post cell group
        if stdp is not None:
            stdp.insert(pre, post)

            if stdp.w_update != "":
                self.has_update_weight = True

            for pre_stdp in pre.stdp_list:
                if not self.compare_obj(pre_stdp, stdp):
                    pre.stdp_list.append(stdp)
            for post_stdp in post.stdp_list:
                if not self.compare_obj(post_stdp, stdp):
                    post.stdp_list.append(stdp)

            if stdp not in pre.stdp_list:
                pre.stdp_list.append(stdp)
            if stdp not in post.stdp_list:
                post.stdp_list.append(stdp)
            self.__stdp_list.append((pre, post, stdp))

    @staticmethod
    def compare_obj(obj1: Union[CellGroup, Synapse, STDP, None], obj2: Union[CellGroup, Synapse, STDP, None]):
        # if same class and same parameter, return True
        if obj1 is None and obj2 is None:
            return True
        if type(obj1) == type(obj2):
            if obj1.params == obj2.params:
                return True
        return False
