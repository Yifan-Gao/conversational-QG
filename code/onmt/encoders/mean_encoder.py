"""Define a minimal encoder."""
from __future__ import division

import torch
import torch.nn as nn

from onmt.encoders.encoder import EncoderBase


class MeanEncoder(nn.Module):
    """A trivial non-recurrent encoder. Simply applies mean pooling.

    Args:
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """

    def __init__(self):
        super(MeanEncoder, self).__init__()

    def forward(self, src, lengths, embeddings, batch_first=False):
        """
        mean encoder
        :param src: [batch_size * max_sent * max_word] or [batch_size * max_word]
        :param lengths: the length of each sample
        :param batch_first: first dim represents batch_size or the last dim
        :param embeddings: embedding matrix
        :return: averaged representation
        """
        if len(src.size()) == 3 and batch_first:
            # because here lengths hasn't been considered
            return NotImplementedError
            # batch_size, max_sent, max_word = src.size()
            # emb = embeddings(src.view(batch_size, -1).unsqueeze(-1))
            # memory_bank = emb.view(batch_size, max_sent, max_word, -1).mean(2)
            # return memory_bank
        elif len(src.size()) == 2 and not batch_first:
            emb = embeddings(src.unsqueeze(-1))
            memory_bank = torch.div(emb.sum(0), lengths.view(src.size(1), 1).type(torch.cuda.FloatTensor))
            return memory_bank
        else:
            raise NotImplementedError
