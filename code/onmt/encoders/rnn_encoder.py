"""Define RNN-based encoders."""
from __future__ import division

import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from onmt.encoders.encoder import EncoderBase
from onmt.utils.rnn_factory import rnn_factory
from onmt.encoders.encoder import PermutationWrapper, PermutationWrapper2D


class RNNEncoder(EncoderBase):
    """ A generic recurrent neural network encoder.

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """

    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, emb_size=300):
        super(RNNEncoder, self).__init__()

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions

        self.rnn, self.no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=emb_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        bidirectional=bidirectional)

    def forward(self, src_emb, lengths=None):
        "See :obj:`EncoderBase.forward()`"
        packed_emb = src_emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Tensor.
            lengths_list = lengths.view(-1).tolist()
            packed_emb = pack(src_emb, lengths_list)

        memory_bank, encoder_final = self.rnn(packed_emb)

        if lengths is not None and not self.no_pack_padded_seq:
            memory_bank = unpack(memory_bank)[0]
        # here encoder_final[0][2,:,:] is the upper layer forward representation
        # here encoder_final[0][3,:,:] is the upper layer backward representation
        return memory_bank, encoder_final, lengths


class InputFeedRNNEncoder(nn.Module):
    """
    InputFeedRNNEncoder = RNNEncoder + embedding
    """
    def __init__(self, rnn_type, encoder_type, enc_layers,
                 hidden_size, dropout=0.0, embeddings=None):
        super(InputFeedRNNEncoder, self).__init__()

        assert embeddings is not None
        self.embeddings = embeddings

        self.rnn_type = rnn_type

        bidirectional = True if encoder_type == 'brnn' else False

        self.rnn_encoder = RNNEncoder(
                rnn_type, bidirectional, enc_layers, hidden_size,
                dropout, embeddings.embedding_size)

    def forward(self, src, lengths=None):
        "See :obj:`EncoderBase.forward()`"
        assert lengths is not None

        wrapped_src = PermutationWrapper(src, lengths, rnn_type=self.rnn_type)
        sorted_src, sorted_lengths = wrapped_src.sort()
        sorted_emb = self.embeddings(sorted_src)

        sorted_memory_bank, sorted_encoder_final, _ = \
            self.rnn_encoder(sorted_emb, sorted_lengths)

        memory_bank, encoder_final = wrapped_src.remap(
            sorted_memory_bank, sorted_encoder_final
        )

        return memory_bank, encoder_final

