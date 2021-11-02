__author__ = "Yuyu Luo"

'''
Define the Encoder of the model
'''

import torch
import torch.nn as nn
from model.SubLayers import MultiHeadAttentionLayer, PositionwiseFeedforwardLayer

class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hid_dim,  # == d_model
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 TOK_TYPES,
                 max_length=128):
        super().__init__()

        self.device = device
        '''
        nn.Embedding: 

        A simple lookup table that stores embeddings of a fixed dictionary and size.
        This module is often used to store word embeddings and retrieve them using indices. 
        The input to the module is a list of indices, and the output is the corresponding word embeddings.
        - num_embeddings (int) – size of the dictionary of embeddings  
        - embedding_dim (int) – the size of each embedding vector 
        '''
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)  # 初始化Embedding

        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        tok_types_num = len(TOK_TYPES.vocab.itos)
        self.tok_types_embedding = nn.Embedding(tok_types_num, hid_dim)

        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask, tok_types, batch_matrix):

        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        src = self.dropout((self.tok_embedding(src) * self.scale) + self.tok_types_embedding(tok_types) + self.pos_embedding(pos))

        # src = [batch size, src len, hid dim]

        for layer in self.layers:
            src, enc_attention = layer(src, src_mask, batch_matrix)

        # src = [batch size, src len, hid dim]
        return src, enc_attention


class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask, batch_matrix):
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, src len]

        # self attention
        _src, _attention = self.self_attention(src, src, src, src_mask, batch_matrix)

        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        # position-wise feedforward
        _src = self.positionwise_feedforward(src)

        # dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]
        # print('EncoderLayer->forward:', src)
        return src, _attention
