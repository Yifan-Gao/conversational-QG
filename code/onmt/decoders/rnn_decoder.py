""" Base Class and function for Decoders """

from __future__ import division
import torch
import torch.nn as nn

import onmt.models.stacked_rnn
from onmt.utils.misc import aeq
from onmt.utils.rnn_factory import rnn_factory
from onmt.decoders.decoder_utils import RNNDecoderState


# class RNNDecoderBase(nn.Module):
#     """
#     Base recurrent attention-based decoder class.
#     Specifies the interface used by different decoder types
#     and required by :obj:`models.NMTModel`.
#
#     Args:
#        rnn_type (:obj:`str`):
#           style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
#        bidirectional_encoder (bool) : use with a bidirectional encoder
#        num_layers (int) : number of stacked layers
#        hidden_size (int) : hidden size of each layer
#        attn_type (str) : see :obj:`onmt.modules.GlobalAttention`
#        coverage_attn (str): see :obj:`onmt.modules.GlobalAttention`
#        context_gate (str): see :obj:`onmt.modules.ContextGate`
#        copy_attn (bool): setup a separate copy attention mechanism
#        dropout (float) : dropout value for :obj:`nn.Dropout`
#        embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
#     """
#
#     def __init__(self, rnn_type, bidirectional_encoder, num_layers,
#                  hidden_size, attn_type="general", attn_func="softmax",
#                  coverage_attn=False, context_gate=None,
#                  copy_attn=False, dropout=0.0, embeddings=None,
#                  reuse_copy_attn=False):
#         super(RNNDecoderBase, self).__init__()
#
#         # Basic attributes.
#         self.decoder_type = 'rnn'
#         self.bidirectional_encoder = bidirectional_encoder
#         self.num_layers = num_layers
#         self.hidden_size = hidden_size
#         self.embeddings = embeddings
#         self.dropout = nn.Dropout(dropout)
#
#         # Build the RNN.
#         self.rnn = self._build_rnn(rnn_type,
#                                    input_size=self._input_size,
#                                    hidden_size=hidden_size,
#                                    num_layers=num_layers,
#                                    dropout=dropout)
#
#         # Set up the context gate.
#         self.context_gate = None
#         if context_gate is not None:
#             self.context_gate = onmt.modules.context_gate_factory(
#                 context_gate, self._input_size,
#                 hidden_size, hidden_size, hidden_size
#             )
#
#         # Set up the standard attention.
#         self._coverage = coverage_attn
#         self.attn = onmt.modules.GlobalAttention(
#             hidden_size, coverage=coverage_attn,
#             attn_type=attn_type, attn_func=attn_func
#         )
#
#         # Set up a separated copy attention layer, if needed.
#         self._copy = False
#         if copy_attn and not reuse_copy_attn:
#             self.copy_attn = onmt.modules.GlobalAttention(
#                 hidden_size, attn_type=attn_type, attn_func=attn_func
#             )
#         if copy_attn:
#             self._copy = True
#         self._reuse_copy_attn = reuse_copy_attn
#
#     def forward(self, tgt, memory_bank, state, memory_lengths=None,
#                 step=None):
#         """
#         Args:
#             tgt (`LongTensor`): sequences of padded tokens
#                  `[tgt_len x batch x nfeats]`.
#             memory_bank (`FloatTensor`): vectors from the encoder
#                  `[src_len x batch x hidden]`.
#             state (:obj:`onmt.models.DecoderState`):
#                  decoder state object to initialize the decoder
#             memory_lengths (`LongTensor`): the padded source lengths
#                 `[batch]`.
#         Returns:
#             (`FloatTensor`,:obj:`onmt.Models.DecoderState`,`FloatTensor`):
#                 * decoder_outputs: output from the decoder (after attn)
#                          `[tgt_len x batch x hidden]`.
#                 * decoder_state: final hidden state from the decoder
#                 * attns: distribution over src at each tgt
#                         `[tgt_len x batch x src_len]`.
#         """
#         # Check
#         assert isinstance(state, RNNDecoderState)
#         # tgt.size() returns tgt length and batch
#         _, tgt_batch, _ = tgt.size()
#         _, memory_batch, _ = memory_bank.size()
#         aeq(tgt_batch, memory_batch)
#         # END
#
#         # Run the forward pass of the RNN.
#         decoder_final, decoder_outputs, attns = self._run_forward_pass(
#             tgt, memory_bank, state, memory_lengths=memory_lengths)
#
#         # Update the state with the result.
#         final_output = decoder_outputs[-1]
#         coverage = None
#         if "coverage" in attns:
#             coverage = attns["coverage"][-1].unsqueeze(0)
#         state.update_state(decoder_final, final_output.unsqueeze(0), coverage)
#
#         # Concatenates sequence of tensors along a new dimension.
#         # NOTE: v0.3 to 0.4: decoder_outputs / attns[*] may not be list
#         #       (in particular in case of SRU) it was not raising error in 0.3
#         #       since stack(Variable) was allowed.
#         #       In 0.4, SRU returns a tensor that shouldn't be stacke
#         if type(decoder_outputs) == list:
#             decoder_outputs = torch.stack(decoder_outputs)
#
#             for k in attns:
#                 if type(attns[k]) == list:
#                     attns[k] = torch.stack(attns[k])
#
#         return decoder_outputs, state, attns
#
#     def init_decoder_state(self, src, memory_bank, encoder_final,
#                            with_cache=False):
#         """ Init decoder state with last state of the encoder """
#         def _fix_enc_hidden(hidden):
#             # The encoder hidden is  (layers*directions) x batch x dim.
#             # We need to convert it to layers x batch x (directions*dim).
#             if self.bidirectional_encoder:
#                 hidden = torch.cat([hidden[0:hidden.size(0):2],
#                                     hidden[1:hidden.size(0):2]], 2)
#             return hidden
#
#         if isinstance(encoder_final, tuple):  # LSTM
#             return RNNDecoderState(self.hidden_size,
#                                    tuple([_fix_enc_hidden(enc_hid)
#                                           for enc_hid in encoder_final]))
#         else:  # GRU
#             return RNNDecoderState(self.hidden_size,
#                                    _fix_enc_hidden(encoder_final))


class InputFeedRNNDecoder(nn.Module):
    """
    Input feeding based decoder. See :obj:`RNNDecoderBase` for options.

    Based around the input feeding approach from
    "Effective Approaches to Attention-based Neural Machine Translation"
    :cite:`Luong2015`

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional_encoder (bool) : use with a bidirectional encoder
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       attn_type (str) : see :obj:`onmt.modules.GlobalAttention`
       coverage_attn (str): see :obj:`onmt.modules.GlobalAttention`
       context_gate (str): see :obj:`onmt.modules.ContextGate`
       copy_attn (bool): setup a separate copy attention mechanism
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """

    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type="general", attn_func="softmax",
                 copy_attn=False, dropout=0.0, embeddings=None,
                 reuse_copy_attn=False):
        super(InputFeedRNNDecoder, self).__init__()

        # Basic attributes.
        self.decoder_type = 'rnn'
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)

        # Build the RNN.
        self.rnn = self._build_rnn(rnn_type,
                                   input_size=self._input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   dropout=dropout)

        # Set up the standard attention.
        self.attn = onmt.modules.GlobalAttention(
            hidden_size, attn_type=attn_type, attn_func=attn_func
        )

        # Set up a separated copy attention layer, if needed.
        self._copy = False
        if copy_attn and not reuse_copy_attn:
            self.copy_attn = onmt.modules.GlobalAttention(
                hidden_size, attn_type=attn_type, attn_func=attn_func
            )
        if copy_attn:
            self._copy = True
        self._reuse_copy_attn = reuse_copy_attn

    def forward(self, tgt, memory_bank, state, memory_lengths=None,
                step=None):
        """
        Args:
            tgt (`LongTensor`): sequences of padded tokens
                 `[tgt_len x batch x nfeats]`.
            memory_bank (`FloatTensor`): vectors from the encoder
                 `[src_len x batch x hidden]`.
            state (:obj:`onmt.models.DecoderState`):
                 decoder state object to initialize the decoder
            memory_lengths (`LongTensor`): the padded source lengths
                `[batch]`.
        Returns:
            (`FloatTensor`,:obj:`onmt.Models.DecoderState`,`FloatTensor`):
                * decoder_outputs: output from the decoder (after attn)
                         `[tgt_len x batch x hidden]`.
                * decoder_state: final hidden state from the decoder
                * attns: distribution over src at each tgt
                        `[tgt_len x batch x src_len]`.
        """
        # Check
        assert isinstance(state, RNNDecoderState)
        # END

        # Run the forward pass of the RNN.
        decoder_final, decoder_outputs, attns = self._run_forward_pass(
            tgt, memory_bank, state, memory_lengths=memory_lengths)

        # Update the state with the result.
        final_output = decoder_outputs[-1]
        coverage = None
        state.update_state(decoder_final, final_output.unsqueeze(0), coverage)

        # Concatenates sequence of tensors along a new dimension.
        # NOTE: v0.3 to 0.4: decoder_outputs / attns[*] may not be list
        #       (in particular in case of SRU) it was not raising error in 0.3
        #       since stack(Variable) was allowed.
        #       In 0.4, SRU returns a tensor that shouldn't be stacke
        if type(decoder_outputs) == list:
            decoder_outputs = torch.stack(decoder_outputs)

            for k in attns:
                if type(attns[k]) == list:
                    attns[k] = torch.stack(attns[k])

        return decoder_outputs, state, attns

    def init_decoder_state(self, encoder_final):
        """ Init decoder state with last state of the encoder """
        def _fix_enc_hidden(hidden):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            if self.bidirectional_encoder:
                hidden = torch.cat([hidden[0:hidden.size(0):2],
                                    hidden[1:hidden.size(0):2]], 2)
            return hidden

        if isinstance(encoder_final, tuple):  # LSTM
            return RNNDecoderState(self.hidden_size,
                                   tuple([_fix_enc_hidden(enc_hid)
                                          for enc_hid in encoder_final]))
        else:  # GRU
            return RNNDecoderState(self.hidden_size,
                                   _fix_enc_hidden(encoder_final))

    def _run_forward_pass(self, tgt, memory_bank, state, memory_lengths=None):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        """
        # Additional args check.
        input_feed = state.input_feed.squeeze(0)
        input_feed_batch, _ = input_feed.size()
        _, tgt_batch = tgt.size()
        aeq(tgt_batch, input_feed_batch)
        # END Additional args check.

        # Initialize local and return variables.
        decoder_outputs = []
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []

        emb = self.embeddings(tgt.unsqueeze(-1))
        assert emb.dim() == 3  # len x batch x embedding_dim

        hidden = state.hidden
        coverage = None

        # Input feed concatenates hidden state with
        # input at every time step.
        for _, emb_t in enumerate(emb.split(1)):
            emb_t = emb_t.squeeze(0)
            decoder_input = torch.cat([emb_t, input_feed], 1)

            rnn_output, hidden = self.rnn(decoder_input, hidden)
            decoder_output, p_attn = self.attn(
                rnn_output,
                memory_bank.transpose(0, 1),
                memory_lengths=memory_lengths)

            decoder_output = self.dropout(decoder_output)
            input_feed = decoder_output

            decoder_outputs += [decoder_output]
            attns["std"] += [p_attn]

            # Run the forward pass of the copy attention layer.
            if self._copy and not self._reuse_copy_attn:
                _, copy_attn = self.copy_attn(decoder_output,
                                              memory_bank.transpose(0, 1))
                attns["copy"] += [copy_attn]
            elif self._copy:
                attns["copy"] = attns["std"]
        # Return result.
        return hidden, decoder_outputs, attns

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        assert not rnn_type == "SRU", "SRU doesn't support input feed! " \
            "Please set -input_feed 0!"
        if rnn_type == "LSTM":
            stacked_cell = onmt.models.stacked_rnn.StackedLSTM
        else:
            stacked_cell = onmt.models.stacked_rnn.StackedGRU
        return stacked_cell(num_layers, input_size,
                            hidden_size, dropout)

    @property
    def _input_size(self):
        """
        Using input feed by concatenating input with attention vectors.
        """
        return self.embeddings.embedding_size + self.hidden_size
