__author__ = "Yuyu Luo"


import pandas as pd
import sqlite3
from dateutil.parser import parse
import json
import py_stringsimjoin as ssj
import re
import os
import time


class ProcessData4Training(object):
    def __init__(self, db_url):
        self.db_url = db_url

    # def is_date(string, fuzzy=False):
    #     """
    #     Return whether the string can be interpreted as a date.
    #
    #     :param string: str, string to check for date
    #     :param fuzzy: bool, ignore unknown tokens in string if True
    #     """
    #     try:
    #         parse(string, fuzzy=fuzzy)
    #         return True
    #
    #     except ValueError:
    #         return False
    #
    # def levenshteinSimilarity(s1, s2):
    #     # Edit Similarity
    #     s1, s2 = s1.lower(), s2.lower()
    #     if len(s1) > len(s2):
    #         s1, s2 = s2, s1
    #
    #     distances = range(len(s1) + 1)
    #     for i2, c2 in enumerate(s2):
    #         distances_ = [i2 + 1]
    #         for i1, c1 in enumerate(s1):
    #             if c1 == c2:
    #                 distances_.append(distances[i1])
    #             else:
    #                 distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
    #         distances = distances_
    #     return 1 - distances[-1] / max(len(s1) + 1, len(s2) + 1)  # [0,1]

    def get_table_columns(self, db_id):
        table_columns = dict()
        '''
        get a list of column names of the tables.
        '''
        try:
            connection = sqlite3.connect(self.db_url + '/' + db_id + '/' + db_id + '.sqlite')
            cursor = connection.execute("SELECT name FROM sqlite_master WHERE type='table';")

            for each_table in cursor.fetchall():
                try:
                    cursor = connection.execute('select * from ' + each_table[0])
                    columns_list = list(map(lambda x: x[0].lower(), cursor.description))  # a list of column names
                    table_columns[each_table[0].lower()] = columns_list
                except:
                    print('table error: ', each_table[0])
                    table_columns[each_table[0].lower()] = []
        except:
            print('db error: ', db_id)

        return table_columns

    def get_values_in_columns(self, db_id, table_id, columns_list, conditions=None):
        '''
        get values in the column

        arg:
            conditions: {
                'numeric_col': 'remove',
                'string_col': {
                    'remove': '50'
                }
            }
        '''
        values_in_columns = dict()

        connection = sqlite3.connect(self.db_url + '/' + db_id + '/' + db_id + '.sqlite')
        cursor = connection.cursor()  # get a cursor
        for col in columns_list:
            try:
                if conditions == None:
                    values_in_columns[col] = list(
                        set([values[0] for values in cursor.execute("select " + col + " from " + table_id)]))
                else:
                    my_list = list(
                        set([values[0] for values in cursor.execute("select " + col + " from " + table_id)]))
                    if all(isinstance(item, int) or isinstance(item, float) or str(item) == '' or str(
                            item) == 'None' for item in my_list) == False:
                        # dont consider numeric col
                        if all(len(str(item)) <= 50 for item in my_list) == True:
                            # remove string column with value length > 50
                            values_in_columns[col] = my_list
                        else:
                            values_in_columns[col] = []
                    else:
                        values_in_columns[col] = []

            except:
                print('error.')

        return values_in_columns
        '''
            {'Team_ID': [1, 2, 3, 4],
             'School_ID': [1, 2, 4, 5]
             }
        '''

    def get_mentioned_values_in_NL_question(self, db_id, table_id, NL_question, db_table_col_val_map):
        '''
        high recall: to find a set of possible columns/vables mentioned in NL_question
        '''
        columns_list = list(db_table_col_val_map[db_id][table_id].keys())
        values = db_table_col_val_map[db_id][table_id]

        # we now only consider 1-gram, 2-gram, and 3-gram of the NL_question.
        NL_tokens = NL_question.split(' ')  # 1-gram
        two_grams, three_grams = [], []
        # 2-gram
        for i in range(len(NL_tokens) - 1):
            two_grams.append(NL_tokens[i] + ' ' + NL_tokens[i + 1])
        # 3-gram
        for i in range(len(NL_tokens) - 2):
            three_grams.append(NL_tokens[i] + ' ' + NL_tokens[i + 1] + ' ' + NL_tokens[i + 2])
        NL_tokens += two_grams
        NL_tokens += three_grams

        A = pd.DataFrame(data=NL_tokens, columns=['name'])
        A['id'] = list(range(len(A)))
        C = pd.DataFrame(data=columns_list, columns=['name'])
        C['id'] = list(range(len(C)))
        cand_col = ssj.edit_distance_join(
            A, C, 'id', 'id', 'name', 'name', 2, l_out_attrs=['name'], r_out_attrs=['name'], show_progress=False
        )
        cand_col = cand_col.sort_values(by=['_sim_score'])
        cand_col = list(cand_col['r_name'])
        candidate_mentioned_col = []
        for i in range(len(cand_col)):
            if cand_col[i] not in candidate_mentioned_col:
                candidate_mentioned_col.append(cand_col[i])
            if len(candidate_mentioned_col) > 10:
                break

        B_value = []
        for k, v in values.items():
            for each_v in v:
                B_value.append([k, each_v])
        B = pd.DataFrame(data=B_value, columns=['col', 'name'])
        B['id'] = list(range(len(B)))
        output_pairs = ssj.edit_distance_join(
            A, B, 'id', 'id', 'name', 'name', 2, l_out_attrs=['name'], r_out_attrs=['name', 'col'], show_progress=False
        )
        output_pairs = output_pairs.sort_values(by=['_sim_score'])
        cand_val = list(zip(output_pairs['r_name'], output_pairs['r_col']))
        candidate_mentioned_val = []
        for i in range(len(cand_val)):
            if cand_val[i][0] not in candidate_mentioned_val:
                candidate_mentioned_val.append(cand_val[i][0])
                if cand_val[i][1] not in candidate_mentioned_col:
                    candidate_mentioned_col.append(cand_val[i][1])
            if len(candidate_mentioned_val) > 10:
                break

        return candidate_mentioned_col, candidate_mentioned_val

    def fill_in_query_template_by_chart_template(self, query):
        '''
        mark = {bar, pie, line, scatter}
        order = {by: x|y, type: desc|asc}
        '''

        query_template = 'mark [T] data [D] encoding x [X] y aggregate [AggFunction] [Y] color [Z] transform filter [F] group [G] bin [B] sort [S] topk [K]'
        query_chart_template = query_template

        query_list = query.lower().split(' ')

        chart_type = query_list[query_list.index('mark') + 1]
        table_name = query_list[query_list.index('data') + 1]

        if 'sort' in query_list:
            # ORDER by X or BY Y?
            xy_axis = query_list[query_list.index('sort') + 1]
            order_xy = '[O]'
            if xy_axis == 'y':
                order_xy = '[Y]'
            elif xy_axis == 'x':
                order_xy = '[X]'
            else:
                order_xy = '[O]'  # other

            if query_list.index('sort') + 2 < len(query_list):
                order_type = query_list[query_list.index('sort') + 2]  # asc / desc
                query_chart_template = query_chart_template.replace('[S]', order_xy + ' ' + order_type)

            else:
                query_chart_template = query_chart_template.replace('[S]', order_xy)

            query_chart_template = query_chart_template.replace('[D]', table_name)

        query_chart_template = query_chart_template.replace('[T]', chart_type)
        query_template = query_template.replace('[D]', table_name)

        return query_template, query_chart_template

    def get_token_types(self, input_source):
        '''
        get token type id (Segment ID)
        '''
        '''
        <nl> Draw a bar chart about the distribution of ACC_Road and the average of Team_ID , and group by attribute ACC_Road, and order in asc by the X-axis. </nl> 
        <template> Visualize BAR SELECT [X] , [AGG(Y)] FROM basketball_match GROUP BY [X] ORDER BY [X] ASC BIN [X] BY [Interval] WHERE [W] </template> 
        <col> Team_ID School_ID Team_Name ACC_Regular_Season ACC_Percent ACC_Home ACC_Road All_Games All_Games_Percent All_Home All_Road All_Neutral </col>
        0 for nl
        1 for template
        2 for col
        3 for val
        '''
        token_types = ''

        for ele in re.findall('<N>.*</N>', input_source)[0].split(' '):
            token_types += ' nl'

        for ele in re.findall('<C>.*</C>', input_source)[0].split(' '):
            token_types += ' template'

        token_types += ' table table'

        for ele in re.findall('<COL>.*</COL>', input_source)[0].split(' '):
            token_types += ' col'

        for ele in re.findall('<VAL>.*</VAL>', input_source)[0].split(' '):
            token_types += ' value'

        token_types += ' table'

        token_types = token_types.strip()
        return token_types

    def process4training(self):
        # process for template
        for each in ['train.csv', 'dev.csv', 'test.csv']:
            df = pd.read_csv('./dataset/' + each)
            data = list()

            for index, row in df.iterrows():

                if str(row['question']) != 'nan':

                    new_row1 = list(row)
                    new_row2 = list(row)

                    query_list = row['vega_zero'].lower().split(' ')
                    table_name = query_list[query_list.index('data') + 1]

                    query_template, query_chart_template = self.fill_in_query_template_by_chart_template(row['vega_zero'])

                    # get a list of mentioned values in the NL question

                    col_names, value_names = self.get_mentioned_values_in_NL_question(
                        row['db_id'], table_name, row['question'], db_table_col_val_map=finding_map
                    )
                    col_names = ' '.join(str(e) for e in col_names)
                    value_names = ' '.join(str(e) for e in value_names)
                    new_row1.append(col_names)
                    new_row1.append(value_names)
                    new_row2.append(col_names)
                    new_row2.append(value_names)

                    new_row1.append(query_template)
                    new_row2.append(query_chart_template)

                    input_source1 = '<N> ' + row[
                        'question'] + ' </N>' + ' <C> ' + query_template + ' </C> ' + '<D> ' + table_name + ' <COL> ' + col_names + ' </COL>' + ' <VAL> ' + value_names + ' </VAL> </D>'
                    input_source1 = ' '.join(input_source1.split())  # Replace multiple spaces with single space

                    input_source2 = '<N> ' + row[
                        'question'] + ' </N>' + ' <C> ' + query_chart_template + ' </C> ' + '<D> ' + table_name + ' <COL> ' + col_names + ' </COL>' + ' <VAL> ' + value_names + ' </VAL> </D>'
                    input_source2 = ' '.join(input_source2.split())  # Replace multiple spaces with single space

                    new_row1.append(input_source1)
                    new_row1.append(row['vega_zero'])

                    new_row2.append(input_source2)
                    new_row2.append(row['vega_zero'])

                    token_types1 = self.get_token_types(input_source1)
                    token_types2 = self.get_token_types(input_source2)
                    new_row1.append(token_types1)
                    new_row2.append(token_types2)

                    data.append(new_row1)
                    data.append(new_row2)
                else:
                    print('nan at ', index)

                if index % 500 == 0:
                    print(round(index / len(df) * 100, 2))

            df_template = pd.DataFrame(data=data, columns=list(df.columns) + ['mentioned_columns', 'mentioned_values',
                                                                              'query_template', 'source', 'labels',
                                                                              'token_types'])
            df_template.to_csv('../dataset/dataset_final/' + each, index=False)

    ### Part 2

    def extract_db_information(self):

        def get_values_in_columns(db_id, table_id, columns_list):
            '''
            get values in the column
            '''
            values_in_columns = dict()

            connection = sqlite3.connect(self.db_url + '/' + db_id + '/' + db_id + '.sqlite')
            cursor = connection.cursor()  # get a cursor
            for col in columns_list:
                try:
                    values_in_columns[col] = list(
                        set([values[0] for values in cursor.execute("select " + col + " from " + table_id)]))
                except:
                    print('error on {0}'.format(db_id))

            return values_in_columns
            '''
                {'Team_ID': [1, 2, 3, 4],
                 'School_ID': [1, 2, 4, 5],
                 'Team_Name': ['Duke', 'Virginia Tech', 'Clemson', 'North Carolina'],
                 }
            '''

        with open('../dataset/db_tables_columns.json') as f:
            data = json.load(f)

        result = []

        for db, tables_data in data.items():
            for table, cols in tables_data.items():
                try:
                    col_val_dict = get_values_in_columns(db, table, cols)
                except Exception as ex:
                    template = "An exception of database -- {0} error occurred."
                    message = template.format(db)
                    print(message)

                for c, v in col_val_dict.items():
                    if len(v) <= 20:
                        for each_v in v:
                            result.append([table, c, each_v])
                    else:
                        result.append([table, c, ''])

        df = pd.DataFrame(data=result, columns=['table', 'column', 'value'])

        df.to_csv('../dataset/database_information.csv', index=False)

if __name__ == "__main__":
    start_time = time.time()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print('It needs about 6 minutes for processing the benchmark datasets')
    time.sleep(2)

    DataProcesser = ProcessData4Training(db_url='../dataset/database')

    # build db-table-column-distinctValue dictionary
    print('build db-table-column-distinctValue dictionary  start ... ...')
    finding_map = dict()

    db_list = os.listdir('../dataset/database/')

    for db in db_list:
        table_cols = DataProcesser.get_table_columns(db)
        finding_map[db] = dict()
        for table, cols in table_cols.items():
            col_val_map = DataProcesser.get_values_in_columns(db, table, cols, conditions='remove')
            finding_map[db][table] = col_val_map

    print('build db-table-column-distinctValue dictionary  end ... ...')

    # process the benchmark dataset for training&testing
    print('process the benchmark dataset for training&testing  start ... ...')
    DataProcesser.process4training()
    print('process the benchmark dataset for training&testing  end ... ...')

    # build 'database_information.csv'
    print("build 'database_information.csv'  start ... ...")
    DataProcesser.extract_db_information()
    print("build 'database_information.csv'  end ... ...")

    print("\n {0} minutes for processing the dataset.".format(round((time.time()-start_time)/60,2)))