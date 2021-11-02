__author__ = "Yuyu Luo"

import re
import json

import torch

def get_candidate_columns(src):
    col_list = re.findall('<col>.*</col>', src)[0].lower().split(' ')
    return col_list[1:-1] # remove <col> </col>

def get_template(src):
    col_list = re.findall('<c>.*</c>', src)[0].lower().split(' ')
    return col_list[1:-1] # remove <template> </template>


def get_all_table_columns(data_file):
    with open(data_file, 'r') as fp:
        data = json.load(fp)
    '''
    return:
    {'chinook_1': {'Album': ['AlbumId', 'Title', 'ArtistId'],
      'Artist': ['ArtistId', 'Name'],
      'Customer': ['CustomerId',
       'FirstName',
    '''
    return data


def get_chart_type(pred_tokens_list):
    return pred_tokens_list[pred_tokens_list.index('mark') + 1]


def get_agg_func(pred_tokens_list):
    return pred_tokens_list[pred_tokens_list.index('aggregate') + 1]


def get_x(pred_tokens_list):
    return pred_tokens_list[pred_tokens_list.index('x') + 1]


def get_y(pred_tokens_list):
    return pred_tokens_list[pred_tokens_list.index('aggregate') + 2]


def guide_decoder_by_candidates(db_id, table_id, trg_field, input_source, table_columns, db_tables_columns_types, topk_ids, topk_tokens,
                                current_token_type, pred_tokens_list):
    '''
    get the current token types (X, Y,...),
    we use the topk tokens from the decoder and the candidate columns to inference the "best" pred_token.
    table_columns: all columns in this table.
    topk_tokens: the top-k candidate predicted tokens
    current_token_type = x|y|groupby-axis|bin x|  if_template:[orderby-axis, order-type, chart_type]
    pred_tokens_list: the predicted tokens list
    '''
    # candidate columns mentioned by the NL query
    candidate_columns = get_candidate_columns(input_source)

    best_token = topk_tokens[0]
    best_id = topk_ids[0]

    if current_token_type == 'x_axis':
        mark_type = get_chart_type(pred_tokens_list)

        if best_token not in table_columns and '(' not in best_token:
            is_in_topk = False
            for tok in topk_tokens:
                if tok in candidate_columns and tok in table_columns:
                    # get column's type
                    if mark_type in ['bar', 'line'] and db_tables_columns_types!=None and db_tables_columns_types[db_id][table_id][tok] != 'numeric':
                        best_token = tok
                        best_id = trg_field.vocab.stoi[best_token]
                        is_in_topk = True
                        break
                    if mark_type == 'point' and db_tables_columns_types!=None and db_tables_columns_types[db_id][table_id][tok] == 'numeric':
                        best_token = tok
                        best_id = trg_field.vocab.stoi[best_token]
                        is_in_topk = True
                        break
                    if mark_type == 'arc' and db_tables_columns_types!=None and db_tables_columns_types[db_id][table_id][tok] != 'numeric':
                        best_token = tok
                        best_id = trg_field.vocab.stoi[best_token]
                        is_in_topk = True
                        break

            if is_in_topk == False and len(candidate_columns) > 0:
                for col in candidate_columns:
                    if col != '':
                        if mark_type in ['bar', 'line'] and db_tables_columns_types != None and db_tables_columns_types[db_id][table_id][col] != 'numeric':
                            best_token = col
                            best_id = trg_field.vocab.stoi[best_token]
                            break

                        if mark_type == 'point' and db_tables_columns_types != None and db_tables_columns_types[db_id][table_id][col] == 'numeric':
                            best_token = col
                            best_id = trg_field.vocab.stoi[best_token]
                            break
                        if mark_type == 'arc' and db_tables_columns_types != None and db_tables_columns_types[db_id][table_id][col] != 'numeric':
                            best_token = col
                            best_id = trg_field.vocab.stoi[best_token]
                            break

    if current_token_type == 'y_axis':
        mark_type = get_chart_type(pred_tokens_list)
        agg_function = get_agg_func(pred_tokens_list)
        selected_x = get_x(pred_tokens_list)

        y = best_token

        if y not in table_columns and y != 'distinct':
            is_in_topk = False
            for tok in topk_tokens:
                if tok in candidate_columns and tok in table_columns:
                    if mark_type in ['bar', 'arc', 'line'] and agg_function == 'count':
                        best_token = tok
                        best_id = trg_field.vocab.stoi[best_token]
                        is_in_topk = True
                        break
                    if mark_type in ['bar', 'arc', 'line'] and agg_function != 'count' and \
                            db_tables_columns_types != None and db_tables_columns_types[db_id][table_id][tok] == 'numeric':
                        best_token = tok
                        best_id = trg_field.vocab.stoi[best_token]
                        is_in_topk = True
                        break
                    if mark_type == 'point' and tok != selected_x:
                        best_token = tok
                        best_id = trg_field.vocab.stoi[best_token]
                        break

            if is_in_topk == False and len(candidate_columns) > 0:
                for col in candidate_columns:
                    if col != '':
                        if mark_type in ['bar', 'arc', 'line'] and agg_function == 'count':
                            best_token = col
                            best_id = trg_field.vocab.stoi[best_token]
                            is_in_topk = True
                            break
                        if mark_type in ['bar', 'arc', 'line'] and agg_function != 'count' and \
                                db_tables_columns_types != None and db_tables_columns_types[db_id][table_id][col] == 'numeric':
                            best_token = col
                            best_id = trg_field.vocab.stoi[best_token]
                            break
                        if mark_type == 'point' and col != selected_x:
                            best_token = col
                            best_id = trg_field.vocab.stoi[best_token]
                            break

        # TODO!
        if (y in table_columns and y not in candidate_columns) and ('(' not in y):
            for tok in topk_tokens:
                if tok in candidate_columns and tok in table_columns:
                    best_token = tok
                    best_id = trg_field.vocab.stoi[best_token]
                    is_in_topk = True
                    break

    if current_token_type == 'z_axis':
        selected_x = get_x(pred_tokens_list)
        selected_y = get_y(pred_tokens_list)

        if best_token not in table_columns or best_token == selected_x or best_token == selected_y:
            is_in_topk = False
            for tok in topk_tokens:
                if tok in candidate_columns and tok in table_columns:
                    # get column's type
                    if selected_x != tok and selected_y != tok and db_tables_columns_types !=None and db_tables_columns_types[db_id][table_id][tok] == 'categorical':
                        best_token = tok
                        best_id = trg_field.vocab.stoi[best_token]
                        is_in_topk = True
                        break

            if is_in_topk == False and len(candidate_columns) > 0:
                for col in candidate_columns:
                    if col != selected_x and col != selected_y and db_tables_columns_types!=None and db_tables_columns_types[db_id][table_id][
                        col] == 'categorical':
                        best_token = col
                        best_id = trg_field.vocab.stoi[best_token]
                        break

        if selected_x == best_token or selected_y == best_token or db_tables_columns_types != None and db_tables_columns_types[db_id][table_id][
            best_token] != 'categorical':
            for tok in topk_tokens:
                if tok in candidate_columns and tok in table_columns:
                    # get column's type
                    if selected_x != tok and selected_y != tok and db_tables_columns_types != None and db_tables_columns_types[db_id][table_id][
                        tok] == 'categorical':
                        best_token = tok
                        best_id = trg_field.vocab.stoi[best_token]
                        break

    if current_token_type == 'topk':  # bin [x] by ..
        is_in_topk = False
        if best_token.isdigit() == False:
            for tok in topk_tokens:
                if tok.isdigit():
                    best_token = tok
                    is_in_topk = True
        if is_in_topk == False:
            best_token = '3'  # default
        best_id = trg_field.vocab.stoi[best_token]

    if current_token_type == 'groupby_axis':
        if best_token != 'x':
            if best_token not in table_columns or db_tables_columns_types != None and db_tables_columns_types[db_id][table_id][best_token] == 'numeric':
                is_in_topk = False
                for tok in topk_tokens:
                    if tok in candidate_columns and tok in table_columns:
                        # get column's type
                        if db_tables_columns_types != None and db_tables_columns_types[db_id][table_id][tok] == 'categorical':
                            best_token = tok
                            best_id = trg_field.vocab.stoi[best_token]
                            is_in_topk = True
                            break

                if is_in_topk == False:
                    best_token = get_x(pred_tokens_list)
                    best_id = trg_field.vocab.stoi[best_token]

    if current_token_type == 'bin_axis':  # bin [x] by ..
        best_token = 'x'
        best_id = trg_field.vocab.stoi[best_token]

    template_list = get_template(input_source)

    if '[t]' not in template_list:  # have the chart template
        if current_token_type == 'chart_type':
            best_token = template_list[template_list.index('mark') + 1]
            best_id = trg_field.vocab.stoi[best_token]

        if current_token_type == 'orderby_axis':
            #   print('Case-3')
            if template_list[template_list.index('sort') + 1] == '[x]':
                best_token = 'x'
                best_id = trg_field.vocab.stoi[best_token]

            elif template_list[template_list.index('sort') + 1] == '[y]':
                best_token = 'y'
                best_id = trg_field.vocab.stoi[best_token]
            else:
                pass
                # print('Let me know this issue!')

        if current_token_type == 'orderby_type':
            best_token = template_list[template_list.index('sort') + 2]
            best_id = trg_field.vocab.stoi[best_token]

    return best_id, best_token


def translate_sentence(sentence, src_field, trg_field, TOK_TYPES, tok_types, model, device, max_len=128):
    model.eval()

    # process the tok_type
    if isinstance(tok_types, str):
        tok_types_ids = tok_types.lower().split(' ')
    else:
        tok_types_ids = [tok_type.lower() for tok_type in tok_types]
    tok_types_ids = [TOK_TYPES.init_token] + tok_types_ids + [TOK_TYPES.eos_token]
    tok_types_ids_indexes = [TOK_TYPES.vocab.stoi[tok_types_id] for tok_types_id in tok_types_ids]
    tok_types_tensor = torch.LongTensor(tok_types_ids_indexes).unsqueeze(0).to(device)

    if isinstance(sentence, str):
        tokens = sentence.lower().split(' ')
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    src_mask = model.make_src_mask(src_tensor)

    # visibility matrix
    batch_visibility_matrix = model.make_visibility_matrix(src_tensor, src_field)

    with torch.no_grad():
        enc_src, enc_attention = model.encoder(src_tensor, src_mask, tok_types_tensor, batch_visibility_matrix)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:, -1].item()

        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:], attention, enc_attention


def translate_sentence_with_guidance(db_id, table_id, sentence, src_field, trg_field, TOK_TYPES, tok_types, SRC, model,
                                     db_tables_columns, db_tables_columns_types, device, max_len=128, show_progress = False):
    model.eval()
    # process the tok_type
    if isinstance(tok_types, str):
        tok_types_ids = tok_types.lower().split(' ')
    else:
        tok_types_ids = [tok_type.lower() for tok_type in tok_types]
    tok_types_ids = [TOK_TYPES.init_token] + \
                    tok_types_ids + [TOK_TYPES.eos_token]
    tok_types_ids_indexes = [TOK_TYPES.vocab.stoi[tok_types_id]
                             for tok_types_id in tok_types_ids]
    tok_types_tensor = torch.LongTensor(
        tok_types_ids_indexes).unsqueeze(0).to(device)

    if isinstance(sentence, str):
        tokens = sentence.lower().split(' ')
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    src_mask = model.make_src_mask(src_tensor)

    # visibility matrix
    batch_visibility_matrix = model.make_visibility_matrix(src_tensor, SRC)

    with torch.no_grad():
        enc_src, enc_attention = model.encoder(src_tensor, src_mask,
                                               tok_types_tensor, batch_visibility_matrix)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    trg_tokens = []

    current_token_type = None
    if show_progress == True:
        print('Show the details in each tokens:')

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        table_columns = []
        try:  # get all columns in a table
            table_columns = db_tables_columns[db_id][table_id]
        except:
            print('[Fail] get all columns in a table')
            table_columns = []

        if current_token_type == 'table_name':
            '''
            only for single table !!!
            '''
            pred_token = table_id
            pred_id = trg_field.vocab.stoi[pred_token]
            if show_progress == True:
                print('-------------------\nCurrent Token Type: Table Name , top-3 tokens: [{}]'.format(
                    current_token_type, pred_token))

        else:
            topk_ids = torch.topk(output, k=5, dim=2, sorted=True).indices[:, -1, :].tolist()[0]
            topk_tokens = [trg_field.vocab.itos[tok_id] for tok_id in topk_ids]

            '''
            apply guide_decoder_by_candidates
            '''
            pred_id, pred_token = guide_decoder_by_candidates(
                db_id, table_id, trg_field, sentence, table_columns, db_tables_columns_types, topk_ids,
                topk_tokens, current_token_type, trg_tokens
            )
            if show_progress == True:
                if current_token_type == None:
                    print('-------------------\nCurrent Token Type: Query Sketch Part , top-3 tokens: [{}]'.format(', '.join(topk_tokens)))
                else:
                    print('-------------------\nCurrent Token Type: {} , original top-3 tokens: [{}] , the final tokens by VisAwareTranslation: {}'.format(current_token_type, ', '.join(topk_tokens), pred_token))

        current_token_type = None

        trg_indexes.append(pred_id)
        trg_tokens.append(pred_token)

        # update the current_token_type and pred_aix here
        # mark bar data apartments encoding x apt_type_code y aggregate count apt_type_code transform group x sort y desc
        if i == 0:
            current_token_type = 'chart_type'

        if i > 1:
            if trg_tokens[-1] == 'data' and trg_tokens[-2] in ['bar', 'arc', 'line', 'point']:
                current_token_type = 'table_name'

        if i > 2:
            if trg_tokens[-1] == 'x' and trg_tokens[-2] == 'encoding':
                current_token_type = 'x_axis'

            if trg_tokens[-1] == 'aggregate' and trg_tokens[-2] == 'y':
                current_token_type = 'aggFunction'

            if trg_tokens[-2] == 'aggregate' and trg_tokens[-1] in ['count', 'sum', 'mean', 'avg', 'max', 'min']:
                current_token_type = 'y_axis'

            if trg_tokens[-3] == 'aggregate' and trg_tokens[-2] in ['count', 'sum', 'mean', 'avg', 'max', 'min'] and \
                    trg_tokens[-1] == 'distinct':
                current_token_type = 'y_axis'

            # mark [T] data photos encoding x [X] y aggregate [AggFunction] [Y] color [Z] transform filter [F] group [G] bin [B] sort [S] topk [K]
            if trg_tokens[-1] == 'color' and trg_tokens[-4] == 'aggregate':
                current_token_type = 'z_axis'

            if trg_tokens[-1] == 'bin':
                current_token_type = 'bin_axis'

            if trg_tokens[-1] == 'group':
                current_token_type = 'groupby_axis'

            if trg_tokens[-1] == 'sort':
                current_token_type = 'orderby_axis'

            if trg_tokens[-2] == 'sort' and trg_tokens[-1] in ['x', 'y']:
                current_token_type = 'orderby_type'

            if trg_tokens[-1] == 'topk':
                current_token_type = 'topk'

        if pred_id == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    return trg_tokens, attention, enc_attention


def postprocessing_group(gold_q_tok, pred_q_tok):
    # 2. checking (and correct) group-by

    # rule: if other part is the same, and only add group-by part, the result should be the same
    if 'group' not in gold_q_tok and 'group' in pred_q_tok:
        groupby_x = pred_q_tok[pred_q_tok.index('group') + 1]
        if ' '.join(pred_q_tok).replace('group ' + groupby_x, '') == ' '.join(gold_q_tok):
            pred_q_tok = gold_q_tok

    return pred_q_tok


def postprocessing(gold_query, pred_query, if_template, src_input):
    try:
        # get the template:
        chart_template = re.findall('<c>.*</c>', src_input)[0]
        chart_template_tok = chart_template.lower().split(' ')

        gold_q_tok = gold_query.lower().split(' ')
        pred_q_tok = pred_query.lower().split(' ')

        # 0. visualize type. if we have the template, the visualization type must be matched.
        if if_template:
            pred_q_tok[pred_q_tok.index('mark') + 1] = gold_q_tok[gold_q_tok.index('mark') + 1]

        # 1. Table Checking. If we focus on single table, must match!!!
        if 'data' in pred_q_tok and 'data' in gold_q_tok:
            pred_q_tok[pred_q_tok.index('data') + 1] = gold_q_tok[gold_q_tok.index('data') + 1]

        pred_q_tok = postprocessing_group(gold_q_tok, pred_q_tok)

        # 3. Order-by. if we have the template, we can checking (and correct) the predicting order-by
        # rule 1: if have the template, order by [x]/[y], trust to the select [x]/[y]
        if 'sort' in gold_q_tok and 'sort' in pred_q_tok and if_template:
            order_by_which_axis = chart_template_tok[chart_template_tok.index('sort') + 1]  # [x], [y], or [o]
            if order_by_which_axis == '[x]':
                pred_q_tok[pred_q_tok.index('sort') + 1] = 'x'
            elif order_by_which_axis == '[y]':
                pred_q_tok[pred_q_tok.index('sort') + 1] = 'y'
            else:
                pass

        elif 'sort' in gold_q_tok and 'sort' not in pred_q_tok and if_template:
            order_by_which_axis = chart_template_tok[chart_template_tok.index('sort') + 1]  # [x], [y], or [o]
            order_type = chart_template_tok[chart_template_tok.index('sort') + 2]

            if 'x' == gold_q_tok[gold_q_tok.index('sort') + 1] or 'y' == gold_q_tok[gold_q_tok.index('sort') + 1]:
                pred_q_tok += ['sort', gold_q_tok[gold_q_tok.index('sort') + 1]]
                if gold_q_tok.index('sort') + 2 < len(gold_q_tok):
                    pred_q_tok += [gold_q_tok[gold_q_tok.index('sort') + 2]]
            else:
                pass

        else:
            pass

        pred_q_tok = postprocessing_group(gold_q_tok, pred_q_tok)

        # 4. checking (and correct) bining
        # rule 1: [interval] bin
        # rule 2: bin by [x]
        if 'bin' in gold_q_tok and 'bin' in pred_q_tok:
            # rule 1
            if_bin_gold, if_bin_pred = False, False

            for binn in ['by time', 'by year', 'by weekday', 'by month']:
                if binn in gold_query:
                    if_bin_gold = binn
                if binn in pred_query:
                    if_bin_pred = binn

            if (if_bin_gold != False and if_bin_pred != False) and (if_bin_gold != if_bin_pred):
                pred_q_tok[pred_q_tok.index('bin') + 3] = if_bin_gold.replace('by ', '')

        if 'bin' in gold_q_tok and 'bin' not in pred_q_tok and 'group' in pred_q_tok:
            # rule 3: group-by x and bin x by time in the bar chart should be the same.
            bin_x = gold_q_tok[gold_q_tok.index('bin') + 1]
            group_x = pred_q_tok[pred_q_tok.index('group') + 1]
            if bin_x == group_x:
                if ''.join(pred_q_tok).replace('group ' + group_x, '') == ''.join(gold_q_tok).replace(
                        'bin ' + bin_x + ' by time', ''):
                    pred_q_tok = gold_q_tok

        # group x | bin x ... count A == count B
        if 'count' in gold_q_tok and 'count' in pred_q_tok:
            if ('group' in gold_q_tok and 'group' in pred_q_tok) or ('bin' in gold_q_tok and 'bin' in pred_q_tok):
                pred_count = pred_q_tok[pred_q_tok.index('count') + 1]
                gold_count = gold_q_tok[gold_q_tok.index('count') + 1]
                if ' '.join(pred_q_tok).replace('count ' + pred_count, 'count ' + gold_count) == ' '.join(gold_q_tok):
                    pred_q_tok = gold_q_tok

    except:
        print('error at post processing')
    return ' '.join(pred_q_tok)