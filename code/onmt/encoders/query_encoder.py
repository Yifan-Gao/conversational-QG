# use answer as query in encoder
from __future__ import division

import torch.nn as nn
import torch.nn.functional as F

from onmt.encoders.encoder import PermutationWrapper
from onmt.encoders.rnn_encoder import RNNEncoder
from onmt.encoders.mean_encoder import MeanEncoder


class QueryEncoder(nn.Module):
    """
    QueryEncoder = RNNEncoder + Encode Query (Answer) + embedding
    """
    def __init__(self, rnn_type, encoder_type, enc_layers,
                 hidden_size, dropout=0.0, embeddings=None):
        super(QueryEncoder, self).__init__()

        assert embeddings is not None
        self.embeddings = embeddings

        self.rnn_type = rnn_type

        bidirectional = True if encoder_type == 'brnn' else False

        self.src_encoder = RNNEncoder(
                rnn_type, bidirectional, enc_layers, hidden_size,
                dropout, embeddings.embedding_size)
        self.answer_encoder = MeanEncoder()

    def forward(self, src, src_lengths, ans, ans_lengths):
        "See :obj:`EncoderBase.forward()`"
        # encode src
        wrapped_src = PermutationWrapper(src, src_lengths, rnn_type=self.rnn_type)
        sorted_src, sorted_src_lengths = wrapped_src.sort()
        sorted_src_emb = self.embeddings(sorted_src)

        sorted_memory_bank, sorted_encoder_final, _ = \
            self.src_encoder(sorted_src_emb, sorted_src_lengths)

        memory_bank, encoder_final = wrapped_src.remap(
            sorted_memory_bank, sorted_encoder_final
        )

        # encode answer
        cur_ans_bank = self.answer_encoder(ans, ans_lengths, self.embeddings,
                                           batch_first=False)

        return memory_bank, encoder_final, cur_ans_bank
