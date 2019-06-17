"""Base class for encoders and generic multi encoders."""

from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from onmt.utils.misc import aeq


class EncoderBase(nn.Module):
    """
    Base encoder class. Specifies the interface used by different encoder types
    and required by :obj:`onmt.Models.NMTModel`.

    .. mermaid::

       graph BT
          A[Input]
          subgraph RNN
            C[Pos 1]
            D[Pos 2]
            E[Pos N]
          end
          F[Memory_Bank]
          G[Final]
          A-->C
          A-->D
          A-->E
          C-->F
          D-->F
          E-->F
          E-->G
    """

    def _check_args(self, src, lengths=None, hidden=None):
        _, n_batch, _ = src.size()
        if lengths is not None:
            n_batch_, = lengths.size()
            aeq(n_batch, n_batch_)

    def forward(self, src, lengths=None):
        """
        Args:
            src (:obj:`LongTensor`):
               padded sequences of sparse indices `[src_len x batch x nfeat]`
            lengths (:obj:`LongTensor`): length of each sequence `[batch]`


        Returns:
            (tuple of :obj:`FloatTensor`, :obj:`FloatTensor`):
                * final encoder state, used to initialize decoder
                * memory bank for attention, `[src_len x batch x hidden]`
        """
        raise NotImplementedError


class PermutationWrapper:
    """Sort the batch according to length, for using RNN pack/unpack"""
    def __init__(self, inputs, length, batch_first=False, rnn_type='LSTM'):
        """
        :param inputs: [seq_length * batch_size]
        :param length: [each sequence length]
        :param batch_first: the first dimension of inputs denotes batch_size or not
        """
        if batch_first:
            inputs = torch.transpose(inputs, 0, 1)
        self.original_inputs = inputs
        self.original_length = length
        self.rnn_type = rnn_type
        self.sorted_inputs = []
        self.sorted_length = []
        # store original position in mapping,
        # e.g. mapping[1] = 5 denotes the tensor which currently in self.sorted_inputs position 5
        # originally locates in position 1 of self.original_inputs
        self.mapping = torch.zeros(self.original_length.size(0)).long().fill_(0)


    def sort(self):
        """
        sort the inputs according to length
        :return: sorted tensor and sorted length
        """
        inputs_list = list(inputs_i.squeeze(1) for inputs_i
                           in torch.split(self.original_inputs, 1, dim=1))
        sorted_inputs = sorted([(length_i.item(), i, inputs_i) for i, (length_i, inputs_i) in
                                enumerate(zip(self.original_length, inputs_list))], reverse=True)
        for i, (length_i, original_idx, inputs_i) in enumerate(sorted_inputs):
            # original_idx: original position in the inputs
            self.mapping[original_idx] = i
            self.sorted_inputs.append(inputs_i)
            self.sorted_length.append(length_i)
        rnn_inputs = torch.stack(self.sorted_inputs, dim=1)
        rnn_length = torch.Tensor(self.sorted_length).type_as(self.original_length)
        return rnn_inputs, rnn_length


    def remap(self, output, state):
        """
        remap the output from RNN to the original input order
        :param output: output from nn.LSTM/GRU, all hidden states
        :param state: final state
        :return: the output and states in original input order
        """
        if self.rnn_type=='LSTM':
            remap_state = tuple(torch.index_select(state_i, 1, self.mapping.cuda())
                                for state_i in state)
        else:
            remap_state = torch.index_select(state, 1, self.mapping.cuda())

        remap_output = torch.index_select(output, 1, self.mapping.cuda())

        return remap_output, remap_state


class PermutationWrapper2D:
    """Permutation Wrapper for 2 levels input like sentence level/word level"""
    def __init__(self, inputs, word_length, sentence_length,
                 batch_first=False, rnn_type='LSTM'):
        """
        :param inputs: 3D input, [batch_size, sentence_seq_length, word_seq_length]
        :param word_length: number of tokens in each sentence
        :param sentence_length: number of sentences in each sample
        :param batch_first: batch_size in first dim of inputs or not
        :param rnn_type: LSTM/GRU
        """
        if batch_first:
            batch_size = inputs.size(0)
        else:
            batch_size = inputs.size(-1)
        self.batch_first = batch_first
        self.original_inputs = inputs
        self.original_word_length = word_length
        self.original_sentence_length = sentence_length
        self.rnn_type = rnn_type
        self.sorted_inputs = []
        self.sorted_length = []
        # store original position in mapping,
        # e.g. mapping[1][3] = 5 denotes the tensor which currently
        # in self.sorted_inputs position 5
        # originally locates in position [1][3] of self.original_inputs
        self.mapping = torch.zeros(batch_size,
                                   sentence_length.max().item()).long().fill_(0)  # (batch_n,sent_n)

    def sort(self):
        """
        sort the inputs according to length
        :return: sorted tensor and sorted length, effective_batch_size: true number of batches
        """
        # first reshape the src into a nested list, remove padded sentences
        inputs_list = list(inputs_i.squeeze(0) for inputs_i
                           in torch.split(self.original_inputs, 1, 0))
        inputs_nested_list = []
        for sent_len_i, sent_i in zip(self.original_sentence_length, inputs_list):
            sent_tmp = list(words_i.squeeze(0) for words_i in torch.split(sent_i, 1, 0))
            inputs_nested_list.append(sent_tmp[:sent_len_i])
        # remove 0 in word_length
        inputs_length_nested_list = []
        for sent_len_i, word_len_i in zip(self.original_sentence_length,
                                          self.original_word_length):
            inputs_length_nested_list.append(word_len_i[:sent_len_i])
        # get a orderedlist, each element: (word_len, sent_idx, word_ijdx, words)
        # sent_idx: i_th example in the batch
        # word_ijdx: j_th sentence sequence in the i_th example
        sorted_inputs = sorted([(sent_len_i[ij].item(), i, ij, word_ij)
                                for i, (sent_i, sent_len_i) in
                                enumerate(zip(inputs_nested_list, inputs_length_nested_list))
                                for ij, word_ij in enumerate(sent_i)], reverse=True)
        # sorted output
        rnn_inputs = []
        rnn_length = []
        for i, word_ij in enumerate(sorted_inputs):
            len_ij, ex_i, sent_ij, words_ij = word_ij
            self.mapping[ex_i, sent_ij] = i + 1  # i+1 because 0 is for empty.
            rnn_inputs.append(words_ij)
            rnn_length.append(len_ij)
        effective_batch_size = len(rnn_inputs)
        rnn_inputs = torch.stack(rnn_inputs, dim=1)
        rnn_length = torch.Tensor(rnn_length).type_as(self.original_word_length)

        return rnn_inputs, rnn_length, effective_batch_size

    def remap(self, output, state):
        """
        remap the output from RNN to the original input order
        :param output: output from nn.LSTM/GRU, all hidden states
        :param state: final state
        :return: the output and states in original input order
        here the returned batch is original_batch * max_len_sent, we need to reshape it further
        """
        # add a all_zero example at the first place
        output_padded = F.pad(output, (0, 0, 1, 0))
        remap_output = torch.index_select(output_padded, 1, self.mapping.view(-1).cuda())
        remap_output = remap_output.view(remap_output.size(0),
                                         self.mapping.size(0), self.mapping.size(1), -1)

        if self.rnn_type == "LSTM":
            h, c = state[0], state[1]
            h_padded = F.pad(h, (0, 0, 1, 0))
            c_padded = F.pad(c, (0, 0, 1, 0))
            remap_h = torch.index_select(h_padded, 1, self.mapping.view(-1).cuda())
            remap_c = torch.index_select(c_padded, 1, self.mapping.view(-1).cuda())
            remap_state = (remap_h.view(remap_h.size(0), self.mapping.size(0), self.mapping.size(1), -1),
                           remap_c.view(remap_c.size(0), self.mapping.size(0), self.mapping.size(1), -1))
        else:
            state_padded = F.pad(state, (0, 0, 1, 0))
            remap_state = torch.index_select(state_padded, 1, self.mapping.view(-1).cuda())
            remap_state = remap_state.view(remap_state.size(0), self.mapping.size(0), self.mapping.size(1), -1)
        return remap_output, remap_state



