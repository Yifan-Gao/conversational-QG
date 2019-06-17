""" Global attention modules (Luong / Bahdanau) """
import torch
import torch.nn as nn
import torch.nn.functional as F

from onmt.modules.sparse_activations import sparsemax
from onmt.utils.misc import aeq, sequence_mask


class QueryAttention(nn.Module):
    """
    Args:
       dim (int): dimensionality of query and key
       attn_type (str): type of attention to use, options [dot,general,mlp]
    """
    def __init__(self, indim, outdim, attn_type="dot", attn_func="softmax"):
        super(QueryAttention, self).__init__()

        self.indim = indim
        self.outdim = outdim
        assert attn_type in ["dot", "general", "mlp"], (
            "Please select a valid attention type.")
        self.attn_type = attn_type
        assert attn_func in ["softmax", "sparsemax"], (
            "Please select a valid attention function.")
        self.attn_func = attn_func

        self.linear_in = nn.Linear(indim, outdim, bias=False)
        self.linear_out = nn.Linear(indim + outdim, outdim, bias=False)

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
        c = torch.bmm(align_vectors, memory_bank)

        # concatenate
        concat_c = torch.cat([c, source], 2).view(batch * target_l,
                                                  self.indim + self.outdim)
        attn_h = self.linear_out(concat_c).view(batch, target_l, self.outdim)

        return attn_h.squeeze(1), align_vectors.squeeze(1)