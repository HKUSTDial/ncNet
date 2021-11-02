__author__ = "Yuyu Luo"

import numpy as np


def create_visibility_matrix(SRC, each_src):
    each_src = np.array(each_src.to('cpu'))

    # find related index
    nl_beg_index = np.where(each_src == SRC.vocab['<n>'])[0][0]
    nl_end_index = np.where(each_src == SRC.vocab['</n>'])[0][0]
    template_beg_index = np.where(each_src == SRC.vocab['<c>'])[0][0]
    template_end_index = np.where(each_src == SRC.vocab['</c>'])[0][0]
    col_beg_index = np.where(each_src == SRC.vocab['<col>'])[0][0]
    col_end_index = np.where(each_src == SRC.vocab['</col>'])[0][0]
    value_beg_index = np.where(each_src == SRC.vocab['<val>'])[0][0]
    value_end_index = np.where(each_src == SRC.vocab['</val>'])[0][0]
    table_name_beg_index = np.where(each_src == SRC.vocab['<d>'])[0][0]
    table_name_end_index = np.where(each_src == SRC.vocab['</d>'])[0][0]

    if SRC.vocab['[d]'] in each_src:
        table_index = np.where(each_src == SRC.vocab['[d]'])[0][0]
    else:
        table_index = -1

    if SRC.vocab['[x]'] in each_src:
        x_index = np.where(each_src == SRC.vocab['[x]'])[0][0]
    else:
        # print('x')
        x_index = -1

    if SRC.vocab['[y]'] in each_src:
        y_index = np.where(each_src == SRC.vocab['[y]'])[0][0]
    else:
        # print('y')
        y_index = -1

    if SRC.vocab['[z]'] in each_src:
        color_index = np.where(each_src == SRC.vocab['[z]'])[0][0]
    else:
        # print('y')
        color_index = -1

    if SRC.vocab['[aggfunction]'] in each_src:
        agg_y_index = np.where(each_src == SRC.vocab['[aggfunction]'])[0][0]
    else:
        agg_y_index = -1
        # print('agg')

    if SRC.vocab['[g]'] in each_src:
        group_index = np.where(each_src == SRC.vocab['[g]'])[0][0]
    else:
        group_index = -1
        # print('xy')

    if SRC.vocab['[b]'] in each_src:
        bin_index = np.where(each_src == SRC.vocab['[b]'])[0][0]
    else:
        bin_index = -1
        # print('xy')

    if SRC.vocab['[s]'] in each_src:
        sort_index = np.where(each_src == SRC.vocab['[s]'])[0][0]
    else:
        sort_index = -1

    if SRC.vocab['[f]'] in each_src:
        where_index = np.where(each_src == SRC.vocab['[f]'])[0][0]
    else:
        where_index = -1
        # print('w')

    if SRC.vocab['[o]'] in each_src:
        other_index = np.where(each_src == SRC.vocab['[o]'])[0][0]
    else:
        other_index = -1
        # print('o')

    if SRC.vocab['[k]'] in each_src:
        topk_index = np.where(each_src == SRC.vocab['[k]'])[0][0]
    else:
        topk_index = -1
        # print('o')

    # init the visibility matrix
    v_matrix = np.zeros(each_src.shape * 2, dtype=int)

    # assign 1 to related cells

    # nl - (nl, template, col, value) self-attention
    v_matrix[nl_beg_index:nl_end_index, :] = 1
    v_matrix[:, nl_beg_index:nl_end_index] = 1

    # col-value self-attention
    v_matrix[col_beg_index:value_end_index, col_beg_index:value_end_index] = 1

    # template self-attention
    v_matrix[template_beg_index:template_end_index,
    template_beg_index:template_end_index] = 1

    # template - col/value self-attention
    # [x]/[y]/[agg(y)]/[o]/[w] <---> col
    # [w] <---> value
    # [c]/[o](order_type)/[i] <---> NL
    if table_index != -1:
        v_matrix[table_index, table_name_beg_index:table_name_end_index] = 1
        v_matrix[table_name_beg_index:table_name_end_index, table_index] = 1

    if x_index != -1:
        v_matrix[x_index, col_beg_index:col_end_index] = 1
        v_matrix[col_beg_index:col_end_index, x_index] = 1
    if y_index != -1:
        v_matrix[y_index, col_beg_index:col_end_index] = 1
        v_matrix[col_beg_index:col_end_index, y_index] = 1
    if color_index != -1:
        v_matrix[color_index, col_beg_index:col_end_index] = 1
        v_matrix[col_beg_index:col_end_index, color_index] = 1

    if agg_y_index != -1:
        v_matrix[agg_y_index, nl_beg_index:nl_end_index] = 1
        v_matrix[nl_beg_index:nl_end_index, agg_y_index] = 1

    if other_index != -1:
        v_matrix[other_index, col_beg_index:col_end_index] = 1
        v_matrix[col_beg_index:col_end_index, other_index] = 1

    if where_index != -1:
        v_matrix[where_index, col_beg_index:col_end_index] = 1
        v_matrix[where_index, value_beg_index:value_end_index] = 1

        v_matrix[col_beg_index:col_end_index, where_index] = 1
        v_matrix[value_beg_index:value_end_index, where_index] = 1

    if group_index != -1:
        v_matrix[group_index, col_beg_index:col_end_index] = 1
        v_matrix[col_beg_index:col_end_index, group_index] = 1

        v_matrix[group_index, nl_beg_index:nl_end_index] = 1
        v_matrix[nl_beg_index:nl_end_index, group_index] = 1

    if bin_index != -1:
        v_matrix[bin_index, col_beg_index:col_end_index] = 1
        v_matrix[col_beg_index:col_end_index, bin_index] = 1

        v_matrix[bin_index, nl_beg_index:nl_end_index] = 1
        v_matrix[nl_beg_index:nl_end_index, bin_index] = 1

    if sort_index != -1:
        v_matrix[sort_index, col_beg_index:col_end_index] = 1
        v_matrix[col_beg_index:col_end_index, sort_index] = 1
    if topk_index != -1:
        v_matrix[topk_index, nl_beg_index:nl_end_index] = 1
        v_matrix[nl_beg_index:nl_end_index, topk_index] = 1

    return v_matrix