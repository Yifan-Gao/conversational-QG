""" Memory attention modules"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from onmt.modules.sparse_activations import sparsemax
from onmt.utils.misc import aeq, sequence_mask, sequence_mask_herd


class MemAttention(nn.Module):
    """
        Args:
       dim (int): dimensionality of query and key
       coverage (bool): use coverage term
       attn_type (str): type of attention to use, options [dot,general,mlp]
    """
    def __init__(self, indim, outdim, coverage=False, attn_type="dot",
                 attn_func="softmax"):
        super(MemAttention, self).__init__()

        self.indim = indim
        self.outdim = outdim
        assert attn_type in ["dot", "general", "mlp"], (
            "Please select a valid attention type.")
        self.attn_type = attn_type
        assert attn_func in ["softmax", "sparsemax"], (
            "Please select a valid attention function.")
        self.attn_func = attn_func

        self.linear_in = nn.Linear(indim, outdim, bias=False)

    def score(self, h_t, h_s):
        """
        Args:
          h_t (`FloatTensor`): sequence of queries `[batch x tgt_len x dim]`
          h_s (`FloatTensor`): sequence of sources `[batch x src_len x dim]`

        Returns:
          :obj:`FloatTensor`:
           raw attention scores (unnormalized) for each src index
          `[batch x tgt_len x src_len]`

        """
        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        aeq(tgt_dim, self.indim)
        aeq(self.outdim, src_dim)

        h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
        h_t_ = self.linear_in(h_t_)
        h_t = h_t_.view(tgt_batch, tgt_len, src_dim)
        h_s_ = h_s.transpose(1, 2)
        # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
        return torch.bmm(h_t, h_s_)


    def forward(self, source, memory_bank, memory_lengths=None, coverage=None):
        raise NotImplementedError


class MemRationaleAttention(MemAttention):
    def forward(self, source, memory_bank, memory_lengths=None, coverage=None):
        """
                Args:
                  source (`FloatTensor`): query vectors `[batch x tgt_len x dim]`
                  memory_bank (`FloatTensor`): source vectors `[batch x src_len x dim]`
                  memory_lengths (`LongTensor`): the source context lengths `[batch]`
                  coverage (`FloatTensor`): None (not supported yet)

                Returns:
                  (`FloatTensor`, `FloatTensor`):

                  * Computed vector `[tgt_len x batch x dim]`
                  * Attention distribtutions for each query
                     `[tgt_len x batch x src_len]`
                """
        if source.dim() == 2:
            source = source.unsqueeze(1)

        batch, source_l, dim = memory_bank.size()
        batch_, target_l, dim_ = source.size()
        aeq(batch, batch_)

        # compute attention scores, as in Luong et al.
        align = self.score(source, memory_bank)

        if memory_lengths is not None:
            mask = sequence_mask(memory_lengths, max_len=align.size(-1))
            mask = mask.unsqueeze(1)  # Make it broadcastable.
            align.masked_fill_(1 - mask, -float('inf'))

        # Softmax or sparsemax to normalize attention weights
        if self.attn_func == "softmax":
            align_vectors = F.softmax(align.view(batch * target_l, source_l), -1)
        else:
            align_vectors = sparsemax(align.view(batch * target_l, source_l), -1)
        align_vectors = align_vectors.view(batch, target_l, source_l)

        # each context vector c_t is the weighted average
        # over all the source hidden states
        # [bs, tgtlen, srclen] x [bs, srclen, hid] -> [bs, tgtlen, hid]
        c = torch.bmm(align_vectors, memory_bank)

        return c.squeeze(1), align_vectors.squeeze(1)


class MemAnswerAttention(MemAttention):
    def forward(self, source, memory_bank, memory_lengths=None, coverage=None):
        # here we do not need to calculate the align
        # because the answer vector is already averaged representations
        if source.dim() == 2:
            source = source.unsqueeze(1)

        batch, source_l, dim = memory_bank.size()
        batch_, target_l, dim_ = source.size()
        aeq(batch, batch_)

        # compute attention scores, as in Luong et al.
        align = self.score(source, memory_bank)

        if memory_lengths is not None:
            mask = sequence_mask(memory_lengths, max_len=align.size(-1))
            mask = mask.unsqueeze(1)  # Make it broadcastable.
            align.masked_fill_(1 - mask, -float('inf'))

        # Softmax or sparsemax to normalize attention weights
        if self.attn_func == "softmax":
            align_vectors = F.softmax(align.view(batch * target_l, source_l), -1)
        else:
            align_vectors = sparsemax(align.view(batch * target_l, source_l), -1)
        align_vectors = align_vectors.view(batch, target_l, source_l)

        # each context vector c_t is the weighted average
        # over all the source hidden states
        c = torch.bmm(align_vectors, memory_bank)

        return c.squeeze(1), align_vectors


class MemQuestionAttention(nn.Module):
    def __init__(self, indim, outdim, coverage=False, attn_type="dot",
                 attn_func="softmax"):
        super(MemQuestionAttention, self).__init__()
        self.indim = indim
        self.outdim = outdim
        assert attn_type in ["dot", "general", "mlp"], (
            "Please select a valid attention type.")
        self.attn_type = attn_type
        assert attn_func in ["softmax", "sparsemax"], (
            "Please select a valid attention function.")
        self.attn_func = attn_func

        # word level
        self.word_linear_in = nn.Linear(indim, outdim, bias=False)
        # self.word_linear_out = nn.Linear(indim + outdim, outdim, bias=False)
        # turn level
        self.turn_linear_in = nn.Linear(indim, outdim, bias=False)

    def word_score(self, h_t, h_s):
        """
        Args:
          h_t (`FloatTensor`): sequence of queries `[batch x tgt_len x dim]`
          h_s (`FloatTensor`): sequence of sources `[batch x src_len x dim]`

        Returns:
          :obj:`FloatTensor`:
           raw attention scores (unnormalized) for each src index
          `[batch x tgt_len x src_len]`

        """
        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        aeq(tgt_dim, self.indim)
        aeq(self.outdim, src_dim)

        h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
        h_t_ = self.word_linear_in(h_t_)
        h_t = h_t_.view(tgt_batch, tgt_len, src_dim)
        h_s_ = h_s.transpose(1, 2)
        # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
        return torch.bmm(h_t, h_s_)

    def turn_score(self, h_t, h_s):
        """
        Args:
          h_t (`FloatTensor`): sequence of queries `[batch x tgt_len x dim]`
          h_s (`FloatTensor`): sequence of sources `[batch x src_len x dim]`

        Returns:
          :obj:`FloatTensor`:
           raw attention scores (unnormalized) for each src index
          `[batch x tgt_len x src_len]`

        """
        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        aeq(tgt_dim, self.indim)
        aeq(self.outdim, src_dim)

        h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
        h_t_ = self.turn_linear_in(h_t_)
        h_t = h_t_.view(tgt_batch, tgt_len, src_dim)
        h_s_ = h_s.transpose(1, 2)
        # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
        return torch.bmm(h_t, h_s_)

    def forward(self, source, memory_bank, memory_lengths=None,
                memory_turns = None, coverage=None):
        # here we implement a hierarchical attention
        if source.dim() == 2:
            source = source.unsqueeze(1)

        batch, source_tl, source_wl, dim = memory_bank.size()
        batch_, target_l, dim_ = source.size()
        aeq(batch, batch_)
        # word level attention
        word_align = self.word_score(source, memory_bank.contiguous()
                           .view(batch, -1, dim))

        # transform align (b, 1, tl * wl) -> (b * tl, 1, wl)
        word_align = word_align.view(batch * source_tl, 1, source_wl)
        if memory_lengths is not None:
            word_mask = sequence_mask_herd(memory_lengths.view(-1), max_len=word_align.size(-1))
            word_mask = word_mask.unsqueeze(1)  # Make it broadcastable.
            word_align.masked_fill_(1 - word_mask, -float('inf'))

        # Softmax or sparsemax to normalize attention weights
        if self.attn_func == "softmax":
            word_align_vectors = F.softmax(word_align.view(batch * source_tl, source_wl), -1)
        else:
            word_align_vectors = sparsemax(word_align.view(batch * source_tl, source_wl), -1)

        # mask the all padded sentences
        sent_pad_mask = memory_lengths.view(-1).eq(0).unsqueeze(1)
        word_align_vectors = torch.mul(word_align_vectors,
                                       (1.0 - sent_pad_mask).type_as(word_align_vectors))
        word_align_vectors = word_align_vectors.view(batch * source_tl, target_l, source_wl)

        # each context vector c_t is the weighted average
        # over all the source hidden states
        cw = torch.bmm(word_align_vectors, memory_bank.view(batch * source_tl, source_wl, -1))
        cw = cw.view(batch, source_tl, -1)
        # concat_cw = torch.cat([cw, source.repeat(1, source_tl, 1)], 2).view(batch*source_tl, -1)
        # attn_hw = self.word_linear_out(concat_cw).view(batch, source_tl, -1)
        # attn_hw = torch.tanh(attn_hw)

        # turn level attention
        turn_align = self.turn_score(source, cw)

        if memory_turns is not None:
            turn_mask = sequence_mask(memory_turns, max_len=turn_align.size(-1))
            turn_mask = turn_mask.unsqueeze(1)  # Make it broadcastable.
            turn_align.masked_fill_(1 - turn_mask, -float('inf'))

        # Softmax or sparsemax to normalize attention weights
        if self.attn_func == "softmax":
            turn_align_vectors = F.softmax(turn_align.view(batch * target_l, source_tl), -1)
        else:
            turn_align_vectors = sparsemax(turn_align.view(batch * target_l, source_tl), -1)
        turn_align_vectors = turn_align_vectors.view(batch, target_l, source_tl)

        # each context vector c_t is the weighted average
        # over all the source hidden states
        ct = torch.bmm(turn_align_vectors, cw)

        return ct.squeeze(1), None


