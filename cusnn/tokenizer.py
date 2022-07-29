from enum import Enum
from typing import List, Union
from .cells import CellGroup
from .value_obj import StateVariable

__all__ = ["euler_method", "rk4_method"]


def tokenize(code: str) -> List[str]:
    token_list = []
    op = ["+", "-", "*", "/", "(", ")", "=", ">", "<", "{", "}", ";", "&", "|", "%"]
    code = code.replace(" ", "").replace("\t", "").replace("\n", "")
    st = 0
    for en, c in enumerate(code):
        if c in op:
            if en - st >= 1:
                token_list.append(code[st:en])
            token_list.append(c)
            st = en + 1
    # 最後の区画
    if len(code) - st >= 1:
        token_list.append(code[st:])

    return token_list


def swap_acceleration(cell_group: CellGroup, code_blocks: list):

    # swap acceleration
    for var in cell_group.s_vars:
        cell_group.allocation.add("float %s_lm;" % var.name)
        cell_group.pre_loop.add("%s_lm = %s[cid];" % (var.name, var.name))
        cell_group.post_loop.add("%s[cid] = %s_lm;" % (var.name, var.name))
        for cb in code_blocks:
            for i, line in enumerate(cb):
                tk_line = tokenize(line)
                for j, tk in enumerate(tk_line):
                    if tk == var.name + "[cid]":
                        tk_line[j] = "%s_lm" % var.name
                cb[i] = "".join(tk_line)
    return "_lm"


def add_cid_to_gm(cell_group: CellGroup, code_blocks: list):
    for var in cell_group.g_vars:
        for cb in code_blocks:
            for i, line in enumerate(cb):
                tk_line = tokenize(line)
                for j, tk in enumerate(tk_line):
                    if tk == var.name:
                        tk_line[j] = "%s[cid]" % var.name
                cb[i] = "".join(tk_line)


def ref_check(var_list: List[StateVariable]):
    var_name = [var.name for var in var_list]
    nr_var = set(var_list)
    r_var = set()    # referenced variables
    # (reference other variable) or (referenced by other equation)
    for var in var_list:
        for tk in var.token_list:
            if tk != var.name and tk in var_name:
                r_var.add(var)
                r_var.add(var_list[var_name.index(tk)])
    return list(r_var), list(nr_var - r_var)


def stdp_swap(stdp_code: str, s_var_name: list):
    token_list = tokenize(stdp_code)
    for i, tk in enumerate(token_list):
        for s_var in s_var_name:
            if tk == s_var+"[cid]":
                token_list[i] = "%s_lm" % s_var
    return "".join(token_list)


def euler_method(cell_group: CellGroup, gm_swap=False):

    var_names = [var.name for var in cell_group.s_vars]
    suffix = "[cid]"

    code_blocks = [list(cell_group.spike_receive),
                   list(cell_group.pre_update),
                   list(cell_group.update),
                   list(cell_group.spike_detection),
                   list(cell_group.spike_event)]

    # tokenize derivative
    for sv in cell_group.s_vars:
        sv.token_list = tokenize(sv.derivative)

    # check dependency of vars
    r_var, nr_var = ref_check(cell_group.s_vars)

    add_cid_to_gm(cell_group, code_blocks)

    # swap acceleration
    if gm_swap:
        suffix = swap_acceleration(cell_group, code_blocks)

    # not referenced variable
    for var in nr_var:
        der = "%s%s += $DT$ *(" % (var.name, suffix)
        for tk in var.token_list:
            der += tk
            if tk in var_names:
                der += suffix
        code_blocks[2].append(der + ");")

    # referenced variable
    for var in r_var:
        der = "%s%s += $DT$ *(" % (var.name, suffix)
        cell_group.allocation.add("float %s_tmp;" % var.name)
        code_blocks[1].append("%s_tmp = %s%s;" % (var.name, var.name, suffix))
        for tk in var.token_list:
            der += tk
            if tk in var_names:
                der += "_tmp"
        code_blocks[2].append(der + ");")

    return code_blocks


def rk4_method(cell_group: CellGroup, gm_swap=False):

    var_names = [var.name for var in cell_group.s_vars]
    nr_k = "k"
    suffix = "[cid]"

    code_blocks = [list(cell_group.spike_receive),
                   list(cell_group.pre_update),
                   list(cell_group.update),
                   list(cell_group.spike_detection),
                   list(cell_group.spike_event)]

    # tokenize derivative
    for sv in cell_group.s_vars:
        sv.token_list = tokenize(sv.derivative)

    # check dependency of vars
    r_var, nr_var = ref_check(cell_group.s_vars)

    add_cid_to_gm(cell_group, code_blocks)

    # swap acceleration
    if gm_swap:
        suffix = swap_acceleration(cell_group, code_blocks)

    # not referenced variable
    if len(r_var) == 0:
        cell_group.allocation.add("float k1, k2, k3, k4;")  # define var k1~k4
    else:
        nr_k = r_var[0].name
    for var in nr_var:
        for i in range(1, 5):
            der = "%s%d = $DT$ *(" % (nr_k, i)
            # rk 4th
            for tk in var.token_list:
                if tk in var_names:
                    if i == 1:
                        der += tk + suffix
                    elif i == 2 or i == 3:
                        der += "(%s%s+%s%d/2)" % (tk, suffix, nr_k, i-1)
                    else:
                        der += "(%s%s+%s3)" % (tk, suffix, nr_k)
                else:
                    der += tk
            code_blocks[2].append(der + ");")
        code_blocks[2].append("%s%s += (%s1 + 2*%s2 + 2*%s3 + %s4) / 6;" % (var.name, suffix, nr_k, nr_k, nr_k, nr_k))

    # referenced variable
    for i in range(1, 5):
        for var in r_var:
            cell_group.allocation.add("float %s%d;" % (var.name, i))     # define var k1~k4
            der = "%s%d = $DT$ *(" % (var.name, i)
            # rk 4th
            for tk in var.token_list:
                if tk in var_names:
                    if i == 1:
                        der += tk + suffix
                    elif i == 2 or i == 3:
                        der += "(%s%s+%s%d/2)" % (tk, suffix, tk, i - 1)
                    else:
                        der += "(%s%s+%s3)" % (tk, suffix, tk)
                else:
                    der += tk
            code_blocks[2].append(der + ");")
        if i == 4:
            for var in r_var:
                code_blocks[2].append("%s%s += (%s1 + 2*%s2 + 2*%s3 + %s4) / 6;" % (var.name, suffix, var.name, var.name, var.name, var.name))

    return code_blocks
