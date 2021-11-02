__author__ = "Yuyu Luo"

import torch
from torchtext.data import Field, TabularDataset, BucketIterator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_vocab(data_dir, db_info, batch_size, max_input_length):

    def tokenizer(text):
        return text.split(' ')

    # def tokenizer_src(text):
    #     return text.split(' ')

    SRC = Field(tokenize=tokenizer,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True,
                batch_first=True)

    TOK_TYPES = Field(tokenize=tokenizer,
                      init_token='<sos>',
                      eos_token='<eos>',
                      lower=True,
                      batch_first=True)

    # TODO data_dir = './Code/dataset/vega_zero/dataset_final/'
    train_data, valid_data, test_data = TabularDataset.splits(
        path=data_dir, format='csv', skip_header=True,
        train='train.csv', validation='dev.csv', test='test.csv',
        fields=[
            ('tvBench_id', None),
            ('db_id', None),
            ('chart', None),
            ('hardness', None),
            ('query', None),
            ('question', None),
            ('vega_zero', None),
            ('mentioned_columns', None),
            ('mentioned_values', None),
            ('query_template', None),
            ('src', SRC),
            ('trg', SRC),
            ('tok_types', TOK_TYPES)
        ])

    # TODO  db_info = './Code/dataset/database_information.csv',
    db_information = TabularDataset(
        path=db_info,
        format='csv',
        skip_header=True,
        fields=[
            ('table', SRC),
            ('column', SRC),
            ('value', SRC)
        ]
    )

    SRC.build_vocab(train_data, valid_data, test_data, db_information, min_freq=2)
    TRG = SRC
    TOK_TYPES.build_vocab(train_data, valid_data, test_data, db_information, min_freq=2)

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data), sort=False,
        batch_size=batch_size,
        device=device)

    return SRC, TRG, TOK_TYPES, batch_size, train_iterator, valid_iterator, test_iterator, max_input_length