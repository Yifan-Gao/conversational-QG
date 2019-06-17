""" coref+copy Generator module """
import torch.nn as nn
import torch
import torch.cuda

import onmt
import onmt.inputters as inputters
from onmt.utils.misc import aeq
from onmt.utils import loss

from onmt.modules.copy_generator import CopyGenerator, CopyGeneratorCriterion, CopyGeneratorLossCompute

class CorefGenerator(CopyGenerator):
    """coref+copy generator"""

    def __init__(self, input_size, tgt_dict, coref_dict):
        super(CorefGenerator, self).__init__(input_size, tgt_dict)
        # self.linear_coref = nn.Linear(input_size, len(coref_dict))
        self.coref_dict = coref_dict
        # find coref_dict vocab corresponding index in tgt_dict
        self.tgt2coref = []
        for coref_token in coref_dict.itos:
            # print(coref_token)
            self.tgt2coref.append(tgt_dict.stoi[coref_token])

    def forward(self, hidden, attn, src_map):
        # CHECKS
        batch_by_tlen, _ = hidden.size()
        batch_by_tlen_, slen = attn.size()
        slen_, batch, cvocab = src_map.size()
        aeq(batch_by_tlen, batch_by_tlen_)
        aeq(slen, slen_)

        # Original probabilities.
        logits = self.linear(hidden)
        logits[:, self.tgt_dict.stoi[inputters.PAD_WORD]] = -float('inf')
        prob = self.softmax(logits)

        # Probability of copying p(z=1) batch.
        p_copy = self.sigmoid(self.linear_copy(hidden))
        # Probibility of not copying: p_{word}(w) * (1 - p(z))
        out_prob = torch.mul(prob, 1 - p_copy.expand_as(prob))
        mul_attn = torch.mul(attn, p_copy.expand_as(attn))
        copy_prob = torch.bmm(mul_attn.view(-1, batch, slen)
                              .transpose(0, 1),
                              src_map.transpose(0, 1)).transpose(0, 1)
        copy_prob = copy_prob.contiguous().view(-1, cvocab)

        # gather coref probs in out_prob
        tgt2coref = torch.Tensor(self.tgt2coref).unsqueeze(0).repeat(out_prob.size(0), 1).long().cuda()
        coref_prob = torch.gather(out_prob, 1, tgt2coref)
        # normalize by 1-p_copy
        coref_prob = coref_prob / ((1 - p_copy).expand_as(coref_prob) + 1e-20)

        # coref_logits = self.linear_coref(hidden)
        # coref_logits[:, self.coref_dict.stoi[inputters.PAD_WORD]] = -float('inf')
        # coref_logits[:, self.coref_dict.stoi[inputters.UNK]] = -float('inf')
        # coref_logits[:, self.coref_dict.stoi[inputters.BOS_WORD]] = -float('inf')
        # coref_logits[:, self.coref_dict.stoi[inputters.EOS_WORD]] = -float('inf')
        # coref_prob = self.softmax(coref_logits)
        # here we manually set nonCoref prob to 0
        # coref_prob[:, self.coref_dict.stoi[inputters.NON_COREF]] = 0

        return torch.cat([out_prob, copy_prob], 1), coref_prob

class CorefVocabCriterion(object):
    """criterion for coref vocab loss"""
    def __init__(self, nonCoref_idx, padding_idx, bos_idx,
                 eos_idx, unk_idx, eps=1e-20):
        self.nonCoref = nonCoref_idx
        self.padding = padding_idx
        self.bos = bos_idx
        self.eos = eos_idx
        self.unk = unk_idx
        self.eps = eps

    def __call__(self, scores, target):
        # compute coref tgt loss
        out = scores.gather(1, target.view(-1, 1)).view(-1) + self.eps
        loss = -out.log().mul((target.ne(self.padding) * \
                              target.ne(self.nonCoref) * \
                              target.ne(self.bos) * \
                              target.ne(self.eos) * \
                              target.ne(self.unk)).float()
                              )
        return loss

class CorefAttnCriterion(object):
    """criterion for coreference attention loss"""
    def __init__(self, padding_idx, eps=1e-20):
        self.padding_idx = padding_idx
        self.eps = eps
    def __call__(self, attn, attn_tgt):
        loss = torch.Tensor([0] * attn.size(1)).type_as(attn)
        for attn_tgt_i in attn_tgt:
            loss[attn_tgt_i[0]] += -(attn[attn_tgt_i[1], attn_tgt_i[0], attn_tgt_i[2]:attn_tgt_i[3]].sum() + self.eps).log()
        return loss

class FlowCriterion(object):
    """criterion for flow loss"""
    def __init__(self, padding_idx, eps=1e-20):
        self.padding_idx = padding_idx
        self.eps = eps
    def __call__(self, attn, attn_tgt):
        loss = -(attn.mul(attn_tgt.unsqueeze(0).float()).sum(dim=2) + self.eps).log()
        return loss

class FlowHistoryCriterion(object):
    """criterion for flow history loss"""
    def __init__(self, padding_idx, eps=1e-20):
        self.padding_idx = padding_idx
        self.eps = eps
    def __call__(self, attn, attn_history):
        loss = attn.mul(attn_history.unsqueeze(0).float()).sum(dim=2)
        return loss

class CorefGeneratorLossCompute(nn.Module):
    """Coref Generator Loss Computation"""
    def __init__(self, generator, tgt_vocab,
                 coref_tgt_vocab,
                 force_copy, normalize_by_length,
                 coref_vocab=True, lambda_coref_vocab=1,
                 coref_attn=True, lambda_coref_attn=1,
                 flow=True, lambda_flow=1,
                 flow_history=True, lambda_flow_history=1,
                 coref_confscore=1,
                 eps=1e-20):
        super(CorefGeneratorLossCompute, self).__init__()
        self.generator = generator
        self.tgt_vocab = tgt_vocab
        self.padding_idx = tgt_vocab.stoi[inputters.PAD_WORD]
        self.force_copy = force_copy
        self.normalize_by_length = normalize_by_length
        self.criterion = CopyGeneratorCriterion(len(tgt_vocab), force_copy,
                                                self.padding_idx)

        self.nonCoref_idx = coref_tgt_vocab.stoi[inputters.NON_COREF]  # index for mask in coref_tgt
        self.coref_bos_idx = coref_tgt_vocab.stoi[inputters.BOS_WORD]
        self.coref_eos_idx = coref_tgt_vocab.stoi[inputters.EOS_WORD]
        self.coref_pad_idx = coref_tgt_vocab.stoi[inputters.PAD_WORD]
        self.coref_unk_idx = coref_tgt_vocab.stoi[inputters.UNK]
        self.coref_vocab = coref_vocab
        self.coref_attn = coref_attn
        self.lambda_coref_vocab = lambda_coref_vocab
        self.lambda_coref_attn = lambda_coref_attn
        self.criterion_coref_vocab = CorefVocabCriterion(self.nonCoref_idx,
                                                         self.coref_pad_idx,
                                                         self.coref_bos_idx,
                                                         self.coref_eos_idx,
                                                         self.coref_unk_idx)
        self.criterion_coref_attn = CorefAttnCriterion(self.padding_idx)

        self.flow = flow
        self.lambda_flow = lambda_flow
        self.flow_history = flow_history
        self.lambda_flow_history = lambda_flow_history
        self.coref_confscore = coref_confscore
        self.flow_criterion = FlowCriterion(self.padding_idx)
        self.flow_history_criterion = FlowHistoryCriterion(self.padding_idx)

    def compute_loss(self, batch, output, attns, normalization):
        """
        Compute the loss. The args must match self._make_shard_state().
        Args:
            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            copy_attn: the copy attention value.
            align: the align info.
        """
        target = batch.tgt[1:].view(-1)
        align = batch.alignment[1:].view(-1)
        copy_attn = attns.get("copy")

        scores, coref_scores = self.generator(self._bottle(output),
                                self._bottle(copy_attn),
                                batch.map)
        loss = self.criterion(scores, align, target)

        # loss for coreference
        coref_vocab_loss_data, coref_attn_loss_data = 0, 0
        coref_confidence = batch.coref_score.unsqueeze(0).repeat(batch.tgt[1:].size(0), 1).view(-1)
        if self.coref_vocab:
            # calculate coref vocab loss
            coref_tgt = batch.coref_tgt[1:].view(-1)
            if self.coref_confscore:
                coref_vocab_loss = (self.criterion_coref_vocab(coref_scores, coref_tgt) * coref_confidence).sum()
            else:
                coref_vocab_loss = (self.criterion_coref_vocab(coref_scores, coref_tgt)).sum()
            if type(coref_vocab_loss) == int:
                coref_vocab_loss = torch.Tensor([coref_vocab_loss]).type_as(coref_scores)
            coref_vocab_loss_data = coref_vocab_loss.data.clone().item()
        if self.coref_attn:
            # calculate coref attention loss
            qa_attn = attns.get("qa")
            if self.coref_confscore:
                coref_attn_loss = (self.criterion_coref_attn(qa_attn, batch.coref_attn_loss) * batch.coref_score).sum()
            else:
                coref_attn_loss = (self.criterion_coref_attn(qa_attn, batch.coref_attn_loss)).sum()
            if type(coref_attn_loss) == int:
                coref_attn_loss = torch.Tensor([coref_attn_loss]).type_as(qa_attn)
            coref_attn_loss_data = coref_attn_loss.data.clone().item()

        # loss for flow tracking
        passage_attn = attns.get("passage")
        flow_loss = self.flow_criterion(passage_attn, batch.sentence_label)
        flow_loss_data = flow_loss.sum().data.clone()
        flow_history_loss = self.flow_history_criterion(passage_attn, batch.history_label)
        flow_history_loss_data = flow_history_loss.sum().data.clone()

        scores_data = scores.data.clone()
        scores_data = inputters.TextDataset.collapse_copy_scores(
            self._unbottle(scores_data, batch.batch_size),
            batch, self.tgt_vocab, batch.dataset.src_vocabs)
        scores_data = self._bottle(scores_data)

        # Correct target copy token instead of <unk>
        # tgt[i] = align[i] + len(tgt_vocab)
        # for i such that tgt[i] == 0 and align[i] != 0
        target_data = target.data.clone()
        correct_mask = target_data.eq(0) * align.data.ne(0)
        correct_copy = (align.data + len(self.tgt_vocab)) * correct_mask.long()
        target_data = target_data + correct_copy

        # Compute sum of perplexities for stats
        loss_data = loss.sum().data.clone()
        stats = self._stats(loss_data, coref_vocab_loss_data, coref_attn_loss_data,
                            len(batch.coref_attn_loss), scores_data, target_data, flow_loss_data, flow_history_loss_data)

        if self.normalize_by_length:
            # Compute Loss as NLL divided by seq length
            # Compute Sequence Lengths
            pad_ix = batch.dataset.fields['tgt'].vocab.stoi[inputters.PAD_WORD]
            tgt_lens = batch.tgt.ne(pad_ix).float().sum(0)
            # Compute Total Loss per sequence in batch
            loss = loss.view(-1, batch.batch_size).sum(0)
            # Divide by length of each sequence and sum
            loss = torch.div(loss, tgt_lens).sum()
        else:
            loss = loss.sum()

        if self.coref_vocab:
            loss = loss + coref_attn_loss * self.lambda_coref_attn
        if self.coref_attn:
            loss = loss + coref_vocab_loss * self.lambda_coref_vocab
        if self.flow:
            loss = loss + flow_loss.sum() * self.lambda_flow
        if self.flow_history:
            loss = loss + flow_history_loss.sum() * self.lambda_flow_history

        loss.div(float(normalization)).backward()

        return stats

    def monolithic_compute_loss(self, batch, output, attns):
        """
        Compute the forward loss for the batch.

        Args:
          batch (batch): batch of labeled examples
          output (:obj:`FloatTensor`):
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict of :obj:`FloatTensor`) :
              dictionary of attention distributions
              `[tgt_len x batch x src_len]`
        Returns:
            :obj:`onmt.utils.Statistics`: loss statistics
        """
        target = batch.tgt[1:].view(-1)
        align = batch.alignment[1:].view(-1)
        copy_attn = attns.get("copy")

        scores, coref_scores = self.generator(self._bottle(output),
                                              self._bottle(copy_attn),
                                              batch.map)
        loss = self.criterion(scores, align, target)

        # loss for coreference
        coref_vocab_loss_data, coref_attn_loss_data = 0, 0
        coref_confidence = batch.coref_score.unsqueeze(0).repeat(batch.tgt[1:].size(0), 1).view(-1)
        if self.coref_vocab:
            # calculate coref vocab loss
            coref_tgt = batch.coref_tgt[1:].view(-1)
            if self.coref_confscore:
                coref_vocab_loss = (self.criterion_coref_vocab(coref_scores, coref_tgt) * coref_confidence).sum()
            else:
                coref_vocab_loss = (self.criterion_coref_vocab(coref_scores, coref_tgt)).sum()
            if type(coref_vocab_loss) == int:
                coref_vocab_loss = torch.Tensor([coref_vocab_loss]).type_as(coref_scores)
            coref_vocab_loss_data = coref_vocab_loss.data.clone().item()
        if self.coref_attn:
            # calculate coref attention loss
            qa_attn = attns.get("qa")
            if self.coref_confscore:
                coref_attn_loss = (self.criterion_coref_attn(qa_attn, batch.coref_attn_loss) * batch.coref_score).sum()
            else:
                coref_attn_loss = (self.criterion_coref_attn(qa_attn, batch.coref_attn_loss)).sum()
            if type(coref_attn_loss) == int:
                coref_attn_loss = torch.Tensor([coref_attn_loss]).type_as(qa_attn)
            coref_attn_loss_data = coref_attn_loss.data.clone().item()

        # loss for flow tracking
        passage_attn = attns.get("passage")
        flow_loss = self.flow_criterion(passage_attn, batch.sentence_label)
        flow_loss_data = flow_loss.sum().data.clone()
        flow_history_loss = self.flow_history_criterion(passage_attn, batch.history_label)
        flow_history_loss_data = flow_history_loss.sum().data.clone()

        scores_data = scores.data.clone()
        scores_data = inputters.TextDataset.collapse_copy_scores(
            self._unbottle(scores_data, batch.batch_size),
            batch, self.tgt_vocab, batch.dataset.src_vocabs)
        scores_data = self._bottle(scores_data)

        # Correct target copy token instead of <unk>
        # tgt[i] = align[i] + len(tgt_vocab)
        # for i such that tgt[i] == 0 and align[i] != 0
        target_data = target.data.clone()
        correct_mask = target_data.eq(0) * align.data.ne(0)
        correct_copy = (align.data + len(self.tgt_vocab)) * correct_mask.long()
        target_data = target_data + correct_copy

        # Compute sum of perplexities for stats
        loss_data = loss.sum().data.clone()
        stats = self._stats(loss_data, coref_vocab_loss_data, coref_attn_loss_data,
                            len(batch.coref_attn_loss), scores_data, target_data, flow_loss_data,
                            flow_history_loss_data)

        return stats

    def _stats(self, loss, corefvocab_loss, corefattn_loss, num_eff_coref, scores, target, flow_loss, flow_history_loss):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target) \
                          .masked_select(non_padding) \
                          .sum() \
                          .item()
        num_non_padding = non_padding.sum().item()
        return onmt.utils.Statistics(loss.item(), flow_loss.item(), flow_history_loss.item(), corefvocab_loss, corefattn_loss,
                                     num_eff_coref, num_non_padding, num_correct)

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))
