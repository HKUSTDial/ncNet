__author__ = "Yuyu Luo"

import numpy as np
import torch
import torch.nn as nn
from model.AttentionForcing import create_visibility_matrix


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Seq2Seq(nn.Module):
    '''
    A transformer-based Seq2Seq model.
    '''
    def __init__(self,
                 encoder,
                 decoder,
                 SRC,
                 src_pad_idx,
                 trg_pad_idx,
                 device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    '''
    The source mask is created by checking where the source sequence is not equal to a <pad> token. 
    It is 1 where the token is not a <pad> token and 0 when it is. 
    It is then unsqueezed so it can be correctly broadcast when applying the mask to the energy, 
    which of shape [batch size, n heads, seq len, seq len].
    '''

    def make_visibility_matrix(self, src, SRC):
        '''
        building the visibility matrix here
        '''
        # src = [batch size, src len]
        batch_matrix = []
        for each_src in src:
            v_matrix = create_visibility_matrix(SRC, each_src)
            n_heads_matrix = [v_matrix] * 8 # TODO: 8 is the number of heads ...
            batch_matrix.append(np.array(n_heads_matrix))
        batch_matrix = np.array(batch_matrix)

        # batch_matrix = [batch size, n_heads, src_len, src_len]
        return torch.tensor(batch_matrix).to(device)

    def make_src_mask(self, src):
        # src = [batch size, src len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        # src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_trg_mask(self, trg):
        # trg = [batch size, trg len]

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

        # trg_pad_mask = [batch size, 1, 1, trg len]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()

        # trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask

        # trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, src, trg, tok_types, SRC):
        # src = [batch size, src len]
        # trg = [batch size, trg len]

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        batch_visibility_matrix = self.make_visibility_matrix(src, SRC)

        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]

        enc_src, enc_attention = self.encoder(src, src_mask, tok_types, batch_visibility_matrix)

        # enc_src = [batch size, src len, hid dim]

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        # output = [batch size, trg len, output dim]
        # attention = [batch size, n heads, trg len, src len]

        return output, attention