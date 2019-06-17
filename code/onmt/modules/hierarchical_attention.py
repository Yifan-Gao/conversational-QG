""" Hierarchical attention modules """
import torch
import torch.nn as nn

from onmt.utils.misc import aeq, sequence_mask, sequence_mask_herd

class HierarchicalAttention(nn.Module):
    """
    Hierarchical attention takes two matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.

    Constructs a unit mapping a query `q` of size `dim`
    and a source matrix `H` of size `n x dim`, to an output
    of size `dim`.


    .. mermaid::

       graph BT
          A[Query]
          subgraph RNN
            C[H 1]
            D[H 2]
            E[H N]
          end
          F[Attn]
          G[Output]
          A --> F
          C --> F
          D --> F
          E --> F
          C -.-> G
          D -.-> G
          E -.-> G
          F --> G

    All models compute the output as
    :math:`c = sum_{j=1}^{SeqLength} a_j H_j` where
    :math:`a_j` is the softmax of a score function.
    Then then apply a projection layer to [q, c].

    However they
    differ on how they compute the attention score.

    * Luong Attention (dot, general):
       * dot: :math:`score(H_j,q) = H_j^T q`
       * general: :math:`score(H_j, q) = H_j^T W_a q`


    * Bahdanau Attention (mlp):
       * :math:`score(H_j, q) = v_a^T tanh(W_a q + U_a h_j)`


    Args:
       dim (int): dimensionality of query and key
       coverage (bool): use coverage term
       attn_type (str): type of attention to use, options [dot,general,mlp]

    """

    def __init__(self, indim, outdim, coverage=False, attn_type="dot",
                 attn_func="softmax"):
        super(HierarchicalAttention, self).__init__()

        self.indim = indim
        self.outdim = outdim
        assert attn_type in ["dot", "general", "mlp"], (
            "Please select a valid attention type.")
        self.attn_type = attn_type
        assert attn_func in ["softmax", "sparsemax"], (
            "Please select a valid attention function.")
        self.attn_func = attn_func

        # Hierarchical attention
        if self.attn_type == "general":
            self.pass_linear_in = nn.Linear(indim, outdim, bias=False)
            self.qa_word_linear_in = nn.Linear(indim, outdim, bias=False)
            self.qa_sent_linear_in = nn.Linear(indim, outdim, bias=False)
            # self.sent_linear_in = nn.Linear(indim, outdim, bias=False)

        self.softmax = nn.Softmax(dim=-1)

    def score(self, h_t, h_s, type):
        """
        Args:
          h_t (`FloatTensor`): sequence of queries `[batch x tgt_len x dim]`
          h_s (`FloatTensor`): sequence of sources `[batch x src_len x dim]`
          type: use word or sent matrix
        Returns:
          :obj:`FloatTensor`:
           raw attention scores (unnormalized) for each src index
          `[batch x tgt_len x src_len]`

        """

        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        aeq(src_dim, tgt_dim)

        h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
        if type == 'qa_word':
            h_t_ = self.qa_word_linear_in(h_t_)
        elif type == 'qa_sent':
            h_t_ = self.qa_sent_linear_in(h_t_)
        elif type == 'pass':
            h_t_ = self.pass_linear_in(h_t_)
        h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
        h_s_ = h_s.transpose(1, 2)
        # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
        return torch.bmm(h_t, h_s_)

    def forward(self, source,
                src_bank, src_lengths,
                qa_sent_bank, qa_sent_lengths,
                qa_word_bank, qa_word_lengths,
                coverage=None):
        """

        Args:
          input (`FloatTensor`): query vectors `[batch x tgt_len x dim]`
          memory_bank (`FloatTensor`): source vectors `[batch x src_len x dim]`
          memory_lengths (`LongTensor`): the source context lengths `[batch]`
          coverage (`FloatTensor`): None (not supported yet)

        Returns:
          (`FloatTensor`, `FloatTensor`):

          * Computed vector `[tgt_len x batch x dim]`
          * Attention distribtutions for each query
             `[tgt_len x batch x src_len]`
        """

        # one step input
        if source.dim() == 2:
            one_step = True
            source = source.unsqueeze(1)
        else:
            one_step = False
        # sent_batch == target_batch
        src_max_len, src_batch, src_dim = src_bank.size()
        qa_word_max_len, qa_word_batch, qa_words_max_len, qa_word_dim = qa_word_bank.size()
        assert src_batch == qa_word_batch
        assert src_dim == qa_word_dim
        qa_sent_max_len, qa_sent_batch, qa_sent_dim = qa_sent_bank.size()
        assert qa_word_batch == qa_sent_batch
        assert qa_words_max_len == qa_sent_max_len
        target_batch, target_l, target_dim = source.size()
        assert src_batch == target_batch

        # reshape for compute word score
        # (qa_word_max_len, qa_word_batch, qa_words_max_len, qa_word_dim) -> transpose
        # (qa_word_batch, qa_word_max_len, qa_words_max_len, qa_word_dim) -> transpose   !!! important, otherwise do not match the src_map
        # (qa_word_batch, qa_words_max_len, qa_word_max_len, qa_word_dim)
        qa_word_bank = qa_word_bank.contiguous().transpose(0, 1).transpose(1, 2).contiguous().view(
            qa_word_batch, qa_words_max_len*qa_word_max_len, qa_word_dim)
        qa_word_align = self.score(source, qa_word_bank, 'qa_word')

        # (src_max_len, src_batch, src_dim) -> (src_batch, src_max_len, src_dim)
        src_bank = src_bank.transpose(0, 1).contiguous()
        src_align = self.score(source, src_bank, 'pass')

        # sentence score
        # (qa_sent_batch, target_l, sent_max_len)
        qa_sent_bank = qa_sent_bank.transpose(0, 1).contiguous()
        qa_sent_align = self.score(source, qa_sent_bank, 'qa_sent')

        # hierarchical qa attention: qa sent * qa word
        qa_hier_align = (qa_word_align.view(qa_word_batch, target_l, qa_words_max_len, qa_word_max_len) * \
            qa_sent_align.unsqueeze(-1)).view(qa_word_batch, target_l, qa_words_max_len * qa_word_max_len)

        # concat src with hier bank
        align = torch.cat([src_align, qa_hier_align], -1)

        # mask
        qa_mask = sequence_mask(qa_word_lengths.view(-1), max_len=qa_word_max_len).view(
            qa_word_batch, qa_words_max_len * qa_word_max_len).unsqueeze(1)
        src_mask = sequence_mask(src_lengths, max_len=src_max_len).view(
            src_batch, src_max_len).unsqueeze(1)
        mask = torch.cat([src_mask, qa_mask], -1)
        align.masked_fill_(1 - mask.cuda(), -float('inf'))

        # qa_hier_align for qa coref loss
        qa_hier_align.masked_fill_(1 - qa_mask.cuda(), -float('inf'))
        qa_hier_vectors = self.softmax(qa_hier_align)

        # src_align for qa coref loss
        src_align.masked_fill_(1 - src_mask.cuda(), -float('inf'))
        src_vectors = self.softmax(src_align)

        ## normalize
        # (word_batch, target_l, words_max_len * word_max_len)
        align_vectors = self.softmax(align)
        # (word_batch, target_l, hid)
        memory_bank = torch.cat([src_bank, qa_word_bank], 1)
        c = torch.bmm(align_vectors, memory_bank)

        return c.squeeze(1), align_vectors.squeeze(1), src_vectors.squeeze(1), qa_hier_vectors.squeeze(1)
