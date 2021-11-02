__author__ = "Yuyu Luo"

'''
This script handles the testing process.
We evaluate the ncNet on the benchmark dataset.
'''

import torch
import torch.nn as nn

from model.VisAwareTranslation import translate_sentence_with_guidance, postprocessing, get_all_table_columns
from model.Model import Seq2Seq
from model.Encoder import Encoder
from model.Decoder import Decoder
from preprocessing.build_vocab import build_vocab

import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test.py')

    parser.add_argument('-model', required=False, default='./save_models/trained_model.pt',
                        help='Path to model weight file')
    parser.add_argument('-data_dir', required=False, default='./dataset/dataset_final/',
                        help='Path to dataset for building vocab')
    parser.add_argument('-db_info', required=False, default='./dataset/database_information.csv',
                        help='Path to database tables/columns information, for building vocab')
    parser.add_argument('-test_data', required=False, default='./dataset/dataset_final/test.csv',
                        help='Path to testing dataset, formatting as csv')
    parser.add_argument('-db_schema', required=False, default='./dataset/db_tables_columns.json',
                        help='Path to database schema file, formatting as json')
    parser.add_argument('-db_tables_columns_types', required=False, default='./dataset/db_tables_columns_types.json',
                        help='Path to database schema file, formatting as json')

    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-max_input_length', type=int, default=128)
    parser.add_argument('-show_progress', required=False, default=False, help='True to show details during decoding')
    opt = parser.parse_args()
    print("the input parameters: ", opt)

    SEED = 1234

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("------------------------------\n| Build vocab start ... | \n------------------------------")
    SRC, TRG, TOK_TYPES, BATCH_SIZE, train_iterator, valid_iterator, test_iterator, my_max_length = build_vocab(
        data_dir=opt.data_dir,
        db_info=opt.db_info,
        batch_size=opt.batch_size,
        max_input_length=opt.max_input_length
    )
    print("------------------------------\n| Build vocab end ... | \n------------------------------")

    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    HID_DIM = 256  # it equals to embedding dimension
    ENC_LAYERS = 3
    DEC_LAYERS = 3
    ENC_HEADS = 8
    DEC_HEADS = 8
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1

    print("------------------------------\n| Build encoder of the ncNet ... | \n------------------------------")
    enc = Encoder(INPUT_DIM,
                  HID_DIM,
                  ENC_LAYERS,
                  ENC_HEADS,
                  ENC_PF_DIM,
                  ENC_DROPOUT,
                  device,
                  TOK_TYPES,
                  my_max_length
                  )

    print("------------------------------\n| Build decoder of the ncNet ... | \n------------------------------")
    dec = Decoder(OUTPUT_DIM,
                  HID_DIM,
                  DEC_LAYERS,
                  DEC_HEADS,
                  DEC_PF_DIM,
                  DEC_DROPOUT,
                  device,
                  my_max_length
                  )

    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    print("------------------------------\n| Build the ncNet structure... | \n------------------------------")
    ncNet = Seq2Seq(enc, dec, SRC, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)  # define the transformer-based ncNet

    print("------------------------------\n| Load the trained ncNet ... | \n------------------------------")
    ncNet.load_state_dict(torch.load(opt.model, map_location=device))


    print("------------------------------\n|          Testing  ...      | \n------------------------------")


    db_tables_columns = get_all_table_columns(opt.db_schema)
    db_tables_columns_types = get_all_table_columns(opt.db_tables_columns_types)

    only_nl_cnt = 0
    only_nl_match = 0
    nl_template_cnt = 0
    nl_template_match = 0

    test_df = pd.read_csv(opt.test_data)

    for index, row in tqdm(test_df.iterrows()):

        try:
            gold_query = row['labels'].lower()

            src = row['source'].lower()

            tok_types = row['token_types']
            table_name = gold_query.split(' ')[gold_query.split(' ').index('data') + 1]
            translation, attention, enc_attention = translate_sentence_with_guidance(
                row['db_id'], table_name, src, SRC, TRG, TOK_TYPES, tok_types, SRC,
                ncNet, db_tables_columns, db_tables_columns_types, device, my_max_length, show_progress=opt.show_progress
            )

            pred_query = ' '.join(translation).replace(' <eos>', '').lower()
            old_pred_query = pred_query

            if '[t]' not in src:
                # with template
                pred_query = postprocessing(gold_query, pred_query, True, src)

                nl_template_cnt += 1

                if ' '.join(gold_query.replace('"', "'").split()) == ' '.join(pred_query.replace('"', "'").split()):
                    nl_template_match += 1
                else:
                    pass


            if '[t]' in src:
                # without template
                pred_query = postprocessing(gold_query, pred_query, False, src)

                only_nl_cnt += 1
                if ' '.join(gold_query.replace('"', "'").split()) == ' '.join(pred_query.replace('"', "'").split()):
                    only_nl_match += 1

                else:
                    pass

        except:
            print('error')

        # if index > 100:
        #     break

    print("========================================================")
    print('ncNet w/o chart template:', only_nl_match / only_nl_cnt)
    print('ncNet with chart template:', nl_template_match / nl_template_cnt)
    print('ncNet overall:', (only_nl_match + nl_template_match) / (only_nl_cnt + nl_template_cnt))
