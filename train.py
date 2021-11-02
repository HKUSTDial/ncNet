__author__ = "Yuyu Luo"

'''
This script handles the training process.
'''


import torch
import torch.nn as nn

from model.Model import Seq2Seq
from model.Encoder import Encoder
from model.Decoder import Decoder
from preprocessing.build_vocab import build_vocab

import numpy as np
import random
import time
import math
import matplotlib.pyplot as plt

import argparse


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        tok_types = batch.tok_types

        optimizer.zero_grad()

        output, _ = model(src, trg[:, :-1], tok_types, SRC)

        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            tok_types = batch.tok_types

            output, _ = model(src, trg[:, :-1], tok_types, SRC)

            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train.py')

    parser.add_argument('-data_dir', required=False, default='./dataset/dataset_final/',
                        help='Path to dataset for building vocab')
    parser.add_argument('-db_info', required=False, default='./dataset/database_information.csv',
                        help='Path to database tables/columns information, for building vocab')
    parser.add_argument('-output_dir', type=str, default='./save_models/')

    parser.add_argument('-epoch', type=int, default=100,
                        help='the number of epoch for training')
    parser.add_argument('-learning_rate', type=float, default=0.0005)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-max_input_length', type=int, default=128)

    # parser.add_argument('-n_head', type=int, default=8)
    # parser.add_argument('-dropout', type=float, default=0.1)
    opt = parser.parse_args()

    ###################################

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    SEED = 1234

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


    print("------------------------------\n| Build vocab start ... | \n------------------------------")
    SRC, TRG, TOK_TYPES, BATCH_SIZE, train_iterator, valid_iterator, test_iterator, my_max_length =  build_vocab(
        data_dir=opt.data_dir,
        db_info=opt.db_info,
        batch_size=opt.batch_size,
        max_input_length=opt.max_input_length
    )
    print("------------------------------\n| Build vocab end ... | \n------------------------------")

    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    HID_DIM = 256 # it equals to embedding dimension
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
    ncNet = Seq2Seq(enc, dec, SRC, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device) # define the transformer-based ncNet

    print("------------------------------\n| Init for training ... | \n------------------------------")
    ncNet.apply(initialize_weights)

    LEARNING_RATE = opt.learning_rate

    optimizer = torch.optim.Adam(ncNet.parameters(), lr=LEARNING_RATE)

    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    N_EPOCHS = opt.epoch
    CLIP = 1

    train_loss_list, valid_loss_list = list(), list()

    best_valid_loss = float('inf')

    print("------------------------------\n| Training start ... | \n------------------------------")

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss = train(ncNet, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(ncNet, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # save the best trained model
        if valid_loss < best_valid_loss:
            print('△○△○△○△○△○△○△○△○\nSave the BEST model!\n△○△○△○△○△○△○△○△○△○')
            best_valid_loss = valid_loss
            torch.save(ncNet.state_dict(), opt.output_dir + 'model_best.pt')

        # save model on each epoch
        print('△○△○△○△○△○△○△○△○\nSave ncNet!\n△○△○△○△○△○△○△○△○△○')
        torch.save(ncNet.state_dict(), opt.output_dir + 'model_' + str(epoch + 1) + '.pt')

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        plt.plot(train_loss_list)
        plt.plot(valid_loss_list)
        plt.show()
