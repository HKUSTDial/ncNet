'''
This file is for convert vis query to the Vega-Lite object
'''


__author__ = "Yuyu Luo"

import json
import pandas

class VegaZero2VegaLite(object):
    def __init__(self):
        pass

    def parse_vegaZero(self, vega_zero):
        self.parsed_vegaZero = {
            'mark': '',
            'data': '',
            'encoding': {
                'x': '',
                'y': {
                    'aggregate': '',
                    'y': ''
                },
                'color': {
                    'z': ''
                }
            },
            'transform': {
                'filter': '',
                'group': '',
                'bin': {
                    'axis': '',
                    'type': ''
                },
                'sort': {
                    'axis': '',
                    'type': ''
                },
                'topk': ''
            }
        }
        vega_zero_keywords = vega_zero.split(' ')

        self.parsed_vegaZero['mark'] = vega_zero_keywords[vega_zero_keywords.index('mark') + 1]
        self.parsed_vegaZero['data'] = vega_zero_keywords[vega_zero_keywords.index('data') + 1]
        self.parsed_vegaZero['encoding']['x'] = vega_zero_keywords[vega_zero_keywords.index('x') + 1]
        self.parsed_vegaZero['encoding']['y']['y'] = vega_zero_keywords[vega_zero_keywords.index('aggregate') + 2]
        self.parsed_vegaZero['encoding']['y']['aggregate'] = vega_zero_keywords[vega_zero_keywords.index('aggregate') + 1]
        if 'color' in vega_zero_keywords:
            self.parsed_vegaZero['encoding']['color']['z'] = vega_zero_keywords[vega_zero_keywords.index('color') + 1]

        if 'topk' in vega_zero_keywords:
            self.parsed_vegaZero['transform']['topk'] = vega_zero_keywords[vega_zero_keywords.index('topk') + 1]

        if 'sort' in vega_zero_keywords:
            self.parsed_vegaZero['transform']['sort']['axis'] = vega_zero_keywords[vega_zero_keywords.index('sort') + 1]
            self.parsed_vegaZero['transform']['sort']['type'] = vega_zero_keywords[vega_zero_keywords.index('sort') + 2]

        if 'group' in vega_zero_keywords:
            self.parsed_vegaZero['transform']['group'] = vega_zero_keywords[vega_zero_keywords.index('group') + 1]

        if 'bin' in vega_zero_keywords:
            self.parsed_vegaZero['transform']['bin']['axis'] = vega_zero_keywords[vega_zero_keywords.index('bin') + 1]
            self.parsed_vegaZero['transform']['bin']['type'] = vega_zero_keywords[vega_zero_keywords.index('bin') + 3]

        if 'filter' in vega_zero_keywords:

            filter_part_token = []
            for each in vega_zero_keywords[vega_zero_keywords.index('filter') + 1:]:
                if each not in ['group', 'bin', 'sort', 'topk']:
                    filter_part_token.append(each)
                else:
                    break

            if 'between' in filter_part_token:
                filter_part_token[filter_part_token.index('between') + 2] = 'and ' + filter_part_token[
                    filter_part_token.index('between') - 1] + ' <='
                filter_part_token[filter_part_token.index('between')] = '>='

            # replace 'and' -- 'or'
            filter_part_token = ' '.join(filter_part_token).split()
            filter_part_token = ['&' if x == 'and' else x for x in filter_part_token]
            filter_part_token = ['|' if x == 'or' else x for x in filter_part_token]

            if '&' in filter_part_token or '|' in filter_part_token:
                final_filter_part = ''
                each_conditions = []
                for i in range(len(filter_part_token)):
                    each = filter_part_token[i]
                    if each != '&' and each != '|':
                        # ’=‘ in SQL --to--> ’==‘ in Vega-Lite
                        if each == '=':
                            each = '=='
                        each_conditions.append(each)
                    if each == '&' or each == '|' or i == len(filter_part_token) - 1:
                        # each = '&' or '|'
                        if 'like' == each_conditions[1]:
                            # only consider this case: '%a%'
                            if each_conditions[2][1] == '%' and each_conditions[2][len(each_conditions[2]) - 2] == '%':
                                final_filter_part += 'indexof(' + 'datum.' + each_conditions[0] + ',"' + \
                                                     each_conditions[2][2:len(each_conditions[2]) - 2] + '") != -1'
                        elif 'like' == each_conditions[2] and 'not' == each_conditions[1]:

                            if each_conditions[3][1] == '%' and each_conditions[3][len(each_conditions[3]) - 2] == '%':
                                final_filter_part += 'indexof(' + 'datum.' + each_conditions[0] + ',"' + \
                                                     each_conditions[3][2:len(each_conditions[3]) - 2] + '") == -1'
                        else:
                            final_filter_part += 'datum.' + ' '.join(each_conditions)

                        if i != len(filter_part_token) - 1:
                            final_filter_part += ' ' + each + ' '
                        each_conditions = []

                self.parsed_vegaZero['transform']['filter'] = final_filter_part

            else:
                # only single filter condition
                self.parsed_vegaZero['transform']['filter'] = 'datum.' + ' '.join(filter_part_token).strip()

        return self.parsed_vegaZero

    def to_VegaLite(self, vega_zero, dataframe=None):
        self.VegaLiteSpec = {
            'bar': {
                "mark": "bar",
                "encoding": {
                    "x": {"field": "x", "type": "nominal"},
                    "y": {"field": "y", "type": "quantitative"}
                }
            },
            'arc': {
                "mark": "arc",
                "encoding": {
                    "color": {"field": "x", "type": "nominal"},
                    "theta": {"field": "y", "type": "quantitative"}
                }
            },
            'line': {
                "mark": "line",
                "encoding": {
                    "x": {"field": "x", "type": "nominal"},
                    "y": {"field": "y", "type": "quantitative"}
                }
            },
            'point': {
                "mark": "point",
                "encoding": {
                    "x": {"field": "x", "type": "quantitative"},
                    "y": {"field": "y", "type": "quantitative"}
                }
            }
        }

        VegaZero = self.parse_vegaZero(vega_zero)

        # assign some vega-zero keywords to the VegaLiteSpec object
        if isinstance(dataframe, pandas.core.frame.DataFrame):
            self.VegaLiteSpec[VegaZero['mark']]['data'] = dict()
            self.VegaLiteSpec[VegaZero['mark']]['data']['values'] = json.loads(dataframe.to_json(orient='records'))

        if VegaZero['mark'] != 'arc':
            self.VegaLiteSpec[VegaZero['mark']]['encoding']['x']['field'] = VegaZero['encoding']['x']
            self.VegaLiteSpec[VegaZero['mark']]['encoding']['y']['field'] = VegaZero['encoding']['y']['y']
            if VegaZero['encoding']['y']['aggregate'] != '' and VegaZero['encoding']['y']['aggregate'] != 'none':
                self.VegaLiteSpec[VegaZero['mark']]['encoding']['y']['aggregate'] = VegaZero['encoding']['y']['aggregate']
        else:
            self.VegaLiteSpec[VegaZero['mark']]['encoding']['color']['field'] = VegaZero['encoding']['x']
            self.VegaLiteSpec[VegaZero['mark']]['encoding']['theta']['field'] = VegaZero['encoding']['y']['y']
            if VegaZero['encoding']['y']['aggregate'] != '' and VegaZero['encoding']['y']['aggregate'] != 'none':
                self.VegaLiteSpec[VegaZero['mark']]['encoding']['theta']['aggregate'] = VegaZero['encoding']['y'][
                    'aggregate']

        if VegaZero['encoding']['color']['z'] != '':
            self.VegaLiteSpec[VegaZero['mark']]['encoding']['color'] = {
                'field': VegaZero['encoding']['color']['z'], 'type': 'nominal'
            }

        # it seems that the group will be performed by VegaLite defaultly, in our cases.
        if VegaZero['transform']['group'] != '':
            pass

        if VegaZero['transform']['bin']['axis'] != '':
            if VegaZero['transform']['bin']['axis'] == 'x':
                self.VegaLiteSpec[VegaZero['mark']]['encoding']['x']['type'] = 'temporal'
                if VegaZero['transform']['bin']['type'] in ['date', 'year', 'week', 'month']:
                    self.VegaLiteSpec[VegaZero['mark']]['encoding']['x']['timeUnit'] = VegaZero['transform']['bin']['type']
                elif VegaZero['transform']['bin']['type'] == 'weekday':
                    self.VegaLiteSpec[VegaZero['mark']]['encoding']['x']['timeUnit'] = 'week'
                else:
                    print('Unknown binning step.')

        if VegaZero['transform']['filter'] != '':
            if 'transform' not in self.VegaLiteSpec[VegaZero['mark']]:
                self.VegaLiteSpec[VegaZero['mark']]['transform'] = [{
                    "filter": VegaZero['transform']['filter']
                }]
            elif 'filter' not in self.VegaLiteSpec[VegaZero['mark']]['transform']:
                self.VegaLiteSpec[VegaZero['mark']]['transform'].append({
                    "filter": VegaZero['transform']['filter']
                })
            else:
                self.VegaLiteSpec[VegaZero['mark']]['transform']['filter'] += ' & ' + VegaZero['transform']['filter']

        if VegaZero['transform']['topk'] != '':
            if VegaZero['transform']['sort']['axis'] == 'x':
                sort_field = VegaZero['encoding']['x']
            elif VegaZero['transform']['sort']['axis'] == 'y':
                sort_field = VegaZero['encoding']['y']['y']
            else:
                print('Unknown sorting field: ', VegaZero['transform']['sort']['axis'])
                sort_field = VegaZero['transform']['sort']['axis']
            if VegaZero['transform']['sort']['type'] == 'desc':
                sort_order = 'descending'
            else:
                sort_order = 'ascending'
            if 'transform' in self.VegaLiteSpec[VegaZero['mark']]:
                current_filter = self.VegaLiteSpec[VegaZero['mark']]['transform'][0]['filter']
                self.VegaLiteSpec[VegaZero['mark']]['transform'][0][
                    'filter'] = current_filter + ' & ' + "datum.rank <= " + str(VegaZero['transform']['topk'])
                self.VegaLiteSpec[VegaZero['mark']]['transform'].insert(0, {
                    "window": [{
                        "field": sort_field,
                        "op": "dense_rank",
                        "as": "rank"
                    }],
                    "sort": [{"field": sort_field, "order": sort_order}]
                })
            else:
                self.VegaLiteSpec[VegaZero['mark']]['transform'] = [
                    {
                        "window": [{
                            "field": sort_field,
                            "op": "dense_rank",
                            "as": "rank"
                        }],
                        "sort": [{"field": sort_field, "order": sort_order}]
                    },
                    {
                        "filter": "datum.rank <= " + str(VegaZero['transform']['topk'])
                    }
                ]

        if VegaZero['transform']['sort']['axis'] != '':
            if VegaZero['transform']['sort']['axis'] == 'x':
                if VegaZero['transform']['sort']['type'] == 'desc':
                    self.VegaLiteSpec[VegaZero['mark']]['encoding']['y']['sort'] = '-x'
                else:
                    self.VegaLiteSpec[VegaZero['mark']]['encoding']['y']['sort'] = 'x'
            else:
                if VegaZero['transform']['sort']['type'] == 'desc':
                    self.VegaLiteSpec[VegaZero['mark']]['encoding']['x']['sort'] = '-y'
                else:
                    self.VegaLiteSpec[VegaZero['mark']]['encoding']['x']['sort'] = 'y'

        return self.VegaLiteSpec[VegaZero['mark']]

