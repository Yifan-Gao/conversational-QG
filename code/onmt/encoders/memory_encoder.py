# implement a memory network like encoder to accepts various types of memory

from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from onmt.encoders.encoder import PermutationWrapper, PermutationWrapper2D
from onmt.encoders.rnn_encoder import RNNEncoder
from onmt.encoders.mean_encoder import MeanEncoder


class MemoryEncoder(nn.Module):
    """
    memory encoder
    rationale memory, question memory, answer memory
    """
    def __init__(self, rnn_type, encoder_type,
                 passage_enc_layers,
                 qa_word_enc_layers,
                 qa_sent_enc_layers,
                 hidden_size, dropout=0.0,
                 embeddings=None, self_attn=1):
        super(MemoryEncoder, self).__init__()

        assert embeddings is not None
        self.embeddings = embeddings

        self.rnn_type = rnn_type

        bidirectional = True if encoder_type == 'brnn' else False

        # passage
        self.passage_encoder = RNNEncoder(
            rnn_type, bidirectional,
            passage_enc_layers, hidden_size,
            dropout, embeddings.embedding_size)
        # qa history
        qa_word_dropout = dropout if qa_word_enc_layers > 1 else 0.0
        qa_sent_dropout = dropout if qa_sent_enc_layers > 1 else 0.0
        self.qa_word_encoder = RNNEncoder(
            rnn_type, bidirectional,
            qa_word_enc_layers, hidden_size,
            qa_word_dropout, embeddings.word_vec_size)  # here the qa history has feature
        # # here for utterance level modeling, we only use unidirectional rnn
        sent_brnn = False
        self.qa_sent_encoder = RNNEncoder(
            rnn_type, sent_brnn,
            qa_sent_enc_layers, hidden_size,
            qa_sent_dropout, hidden_size)

        self.self_attn = self_attn
        if self.self_attn:
            # weight for self attention
            self.selfattn_ws = nn.Linear(hidden_size, hidden_size, bias=False)
            self.selfattn_wf = nn.Linear(hidden_size * 2, hidden_size)
            self.selfattn_wg = nn.Linear(hidden_size * 2, hidden_size)
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, src, src_lengths,
                qa, qa_sent_lengths, qa_word_lengths):
        # src
        wrapped_src = PermutationWrapper(src, src_lengths, rnn_type=self.rnn_type)
        sorted_src, sorted_src_lengths = wrapped_src.sort() # sort
        sorted_src_emb = self.embeddings(sorted_src) # get embedding
        sorted_src_bank, sorted_src_state, _ = \
            self.passage_encoder(sorted_src_emb, sorted_src_lengths) # encode
        src_bank, src_state = wrapped_src.remap(sorted_src_bank, sorted_src_state) # remap

        src_bank_final = src_bank

        if self.self_attn:
            # add gated self attention module like
            # [EMNLP18] Paragraph-level Neural Question Generation with Maxout Pointer and Gated Self-attention Networks
            src_maxlen, src_bs, src_h = src_bank.size()
            # [src_maxlen, src_bs, src_h] -> [src_bs, src_maxlen, src_h]
            src_bank_t = src_bank.transpose(0, 1).contiguous()
            src_bank_all = self.selfattn_ws(src_bank_t.view(-1, src_h)).view(src_bs, src_maxlen, src_h)
            # [src_bs, src_maxlen, src_h] * [src_bs, src_h, src_maxlen] -> [src_bs, src_maxlen, src_maxlen]
            src_score = torch.bmm(src_bank_all, src_bank_t.transpose(-2, -1))

            # here we define a sequence_mask_2d to mask 2d paddings
            def sequence_mask2d(lengths, max_len=None):
                """
                Creates a 2d boolean mask from sequence lengths.
                """
                batch_size = lengths.numel()
                max_len = max_len or lengths.max()
                mask1d = (torch.arange(0, max_len)
                          .type_as(lengths)
                          .repeat(batch_size, 1)
                          .lt(lengths.unsqueeze(1)))
                mask2d_inv = (1 - mask1d.unsqueeze(-1)) * (1 - mask1d.unsqueeze(1))
                # mask2d = mask1d.unsqueeze(1).repeat(1,max_len, 1) + mask2d_inv
                mask2d = mask1d.unsqueeze(-1) * mask1d.unsqueeze(1)
                return mask2d + mask2d_inv

            src_mask = sequence_mask2d(src_lengths, max_len=src_maxlen)
            src_score.masked_fill_(1 - src_mask.cuda(), -float('inf'))
            src_weight = self.softmax(src_score)
            # src_weight.masked_fill_(1 - src_mask.unsqueeze(-1).cuda(), 0.0)
            src_selfattn_s = torch.bmm(src_weight, src_bank_t)
            # gating mechanism
            src_selfattn_f = torch.tanh(
                self.selfattn_wf(torch.cat([src_bank_t, src_selfattn_s], dim=-1).view(-1, src_h * 2)).view(src_bs,
                                                                                                           src_maxlen,
                                                                                                           src_h))
            src_selfattn_g = torch.sigmoid(
                self.selfattn_wg(torch.cat([src_bank_t, src_selfattn_s], dim=-1).view(-1, src_h * 2)).view(src_bs,
                                                                                                           src_maxlen,
                                                                                                           src_h))
            src_selfattn_bank = src_selfattn_g * src_selfattn_f + (1 - src_selfattn_g) * src_bank_t
            src_selfattn_bank = src_selfattn_bank.transpose(0, 1).contiguous()

            src_bank_final = src_selfattn_bank

        # memory_qa
        ## word level
        wrapped_qa_word = PermutationWrapper2D(qa, qa_word_lengths, qa_sent_lengths,
                                               batch_first=True,
                                               rnn_type=self.rnn_type)
        sorted_qa_word, sorted_qa_word_lengths, sorted_qa_bs = wrapped_qa_word.sort() # sort
        # [feat, bs, len] -> [len, bs, feat]
        sorted_qa_word_emb = self.embeddings(sorted_qa_word)  # get embedding
        sorted_qa_word_bank, sorted_qa_word_state, _ = \
            self.qa_word_encoder(sorted_qa_word_emb, sorted_qa_word_lengths)
        qa_word_bank, qa_word_state = wrapped_qa_word.remap(sorted_qa_word_bank, sorted_qa_word_state)

        ## sentence level
        _, bs, sentlen, hid = qa_word_state[0].size()
        # [2n, bs, sentlen, hid] -> [sentlen, bs, 2n, hid] -> [sentlen, bs, hid * 2n]
        qa_sent_emb = qa_word_state[0].transpose(0, 2)[:, :, -2:, :].contiguous().view(sentlen, bs, -1)
        tmp = torch.cat((qa_word_state[0][-2,:,:,:], qa_word_state[0][-1,:,:,:]), dim=-1).transpose(0, 1).contiguous()
        assert torch.equal(qa_sent_emb.data, tmp.data)  # test code
        wrapped_qa_sent = PermutationWrapper(qa_sent_emb, qa_sent_lengths, rnn_type=self.rnn_type)
        sorted_qa_sent_emb, sorted_qa_sent_lengths = wrapped_qa_sent.sort()
        sorted_qa_sent_bank, sorted_qa_sent_state, _ = \
            self.qa_sent_encoder(sorted_qa_sent_emb, sorted_qa_sent_lengths)
        qa_sent_bank, qa_sent_state = wrapped_qa_sent.remap(sorted_qa_sent_bank, sorted_qa_sent_state)

        # here we try use sentence last hid as state
        qa_sent_state = tuple(hid.repeat(2, 1, 1) for hid in qa_sent_state)

        return src_bank_final, qa_sent_bank, qa_word_bank, qa_sent_state