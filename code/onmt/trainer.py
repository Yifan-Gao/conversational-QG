"""
    This is the loadable seq2seq trainer library that is
    in charge of training details, loss compute, and statistics.
    See train.py for a use case of this library.

    Note: To make this a general library, we implement *only*
          mechanism things here(i.e. what to do), and leave the strategy
          things to users(i.e. how to do it). Also see train.py(one of the
          users of this library) for the strategy things we do.
"""

from __future__ import division

import torch

import onmt.inputters as inputters
import onmt.utils

from onmt.utils.logging import logger


def build_trainer(opt, device_id, model, fields,
                  optim, data_type, model_saver=None):
    """
    Simplify `Trainer` creation based on user `opt`s*

    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """
    train_loss = onmt.utils.loss.build_loss_compute(
        model, fields,  opt)
    valid_loss = onmt.utils.loss.build_loss_compute(
        model, fields, opt, train=False)

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches
    norm_method = opt.normalization
    grad_accum_count = opt.accum_count

    report_manager = onmt.utils.build_report_manager(opt)
    trainer = onmt.Trainer(model, train_loss, valid_loss, optim, trunc_size,
                           shard_size, data_type, norm_method,
                           grad_accum_count, report_manager,
                           model_saver=model_saver)
    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self, model, train_loss, valid_loss, optim,
                 trunc_size=0, shard_size=32, data_type='text',
                 norm_method="sents", grad_accum_count=1,
                 report_manager=None, model_saver=None):
        # Basic attributes.
        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.data_type = data_type
        self.norm_method = norm_method
        self.grad_accum_count = grad_accum_count
        self.report_manager = report_manager
        self.model_saver = model_saver

        assert grad_accum_count > 0
        if grad_accum_count > 1:
            assert(self.trunc_size == 0), \
                """To enable accumulated gradients,
                   you must disable target sequence truncating."""

        # Set model in training mode.
        self.model.train()

    def train(self, data_iter, train_steps, valid_steps):
        """
        The main training loops.
        by iterating over training data (i.e. `train_iter_fct`)
        and running validation (i.e. iterating over `valid_iter_fct`

        Args:
            train_iter_fct(function): a function that returns the train
                iterator. e.g. something like
                train_iter_fct = lambda: generator(*args, **kwargs)
            valid_iter_fct(function): same as train_iter_fct, for valid data
            train_steps(int):
            valid_steps(int):
            save_checkpoint_steps(int):

        Return:
            None
        """
        logger.info('Start training...')

        step = self.optim._step + 1

        total_stats = onmt.utils.Statistics()
        report_stats = onmt.utils.Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        while step <= train_steps:
            for i, batch in enumerate(data_iter("train")):
                self._gradient_accumulation(
                    batch, batch.batch_size, total_stats,
                    report_stats)

                report_stats = self._maybe_report_training(
                    step, train_steps,
                    self.optim.learning_rate,
                    report_stats)

                if step % valid_steps == 0:
                    torch.cuda.empty_cache()
                    valid_stats = self.validate(data_iter("valid"))
                    self._report_step(self.optim.learning_rate,
                                      step, valid_stats=valid_stats)

                self._maybe_save(step)
                step += 1
                if step > train_steps:
                    break
        return total_stats

    def _comb_src_qa_map(self, batch):
        # do make_src & make_qa here:
        src_qa_map = torch.cat([batch.src_map, batch.qa_map.view(batch.batch_size, -1)], -1)
        batch.map = torch.zeros(
            src_qa_map.size(-1),
            batch.batch_size,
            int(src_qa_map.max().item()) + 1
        ).type_as(batch.qa_map)
        for i, tokens in enumerate(src_qa_map.split(1, 0)):
            for j, token in enumerate(tokens.squeeze(0).split(1, 0)):
                if int(token.item()) >= 0:
                    batch.map[j, i, int(token.item())] = 1

    def _coref_attn_target(self, batch):
        # create the target for coreference attention
        qawordlen = batch.qa[0].size(2)
        batch.coref_attn_loss = []
        batch_coref_attn = batch.coref_attn.split(1, 0)
        batch_coref_tgt = batch.coref_tgt.split(1, 1)
        for idx, (coref_attn_i, coref_tgt_i) in enumerate(zip(batch_coref_attn, batch_coref_tgt)):
            coref_attn_i = coref_attn_i.squeeze(0)
            coref_tgt_i = coref_tgt_i.squeeze(-1)
            if not coref_attn_i[0].equal(torch.cuda.LongTensor([-1])):
                # find tgt idx
                tgt_max_idx = coref_tgt_i.argmax()
                # transform the span start end position
                coref_start = coref_attn_i[0] * qawordlen + coref_attn_i[1]
                coref_end = coref_attn_i[0] * qawordlen + coref_attn_i[2]
                # batch_idx, position-in-tgt, span_start, span_end
                # we use teacher forcing so 'tgt_max_idx - 1'
                batch.coref_attn_loss.append([idx, tgt_max_idx - 1, coref_start, coref_end])


    def validate(self, valid_iter):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()

        stats = onmt.utils.Statistics()

        for batch in valid_iter:
            # mem_rtl, mem_rtl_length = batch.src[0], batch.src[1]
            # mem_ans, mem_ans_turn, mem_ans_length = \
            #     batch.memory_answer[0], batch.memory_answer[1], batch.memory_answer[2]
            # mem_que, mem_que_turn, mem_que_length = \
            #     batch.memory_question[0], batch.memory_question[1], batch.memory_question[2]
            # cur_ans, cur_ans_length = batch.current_answer[0], batch.current_answer[1]
            #
            # tgt = batch.tgt
            #
            # # F-prop through the model.
            # outputs, attns, _ = \
            #     self.model(mem_rtl, mem_rtl_length,
            #                mem_ans, mem_ans_turn, mem_ans_length,
            #                mem_que, mem_que_turn, mem_que_length,
            #                cur_ans, cur_ans_length,
            #                tgt)

            self._comb_src_qa_map(batch)

            self._coref_attn_target(batch)

            outputs, attns, _ = self._forward_prop(batch)

            # Compute loss.
            batch_stats = self.valid_loss.monolithic_compute_loss(
                batch, outputs, attns)

            # Update statistics.
            stats.update(batch_stats)

            torch.cuda.empty_cache()

        # Set model back to training mode.
        self.model.train()

        return stats

    # def _gradient_accumulation_backup(self, batch, normalization, total_stats,
    #                            report_stats):
    #     target_size = batch.tgt.size(0)
    #     # Truncated BPTT: reminder not compatible with accum > 1
    #     if self.trunc_size:
    #         trunc_size = self.trunc_size
    #     else:
    #         trunc_size = target_size
    #
    #     dec_state = None
    #     mem_rtl, mem_rtl_length = batch.src[0], batch.src[1]
    #     mem_ans, mem_ans_turn, mem_ans_length = batch.memory_answer[0], batch.memory_answer[1], batch.memory_answer[2]
    #     mem_que, mem_que_turn, mem_que_length = batch.memory_question[0], batch.memory_question[1], batch.memory_question[2]
    #     cur_ans, cur_ans_length = batch.current_answer[0], batch.current_answer[1]
    #
    #     tgt_outer = batch.tgt
    #
    #     for j in range(0, target_size-1, trunc_size):
    #         # 1. Create truncated target.
    #         tgt = tgt_outer[j: j + trunc_size]
    #
    #         # 2. F-prop all but generator.
    #         if self.grad_accum_count == 1:
    #             self.model.zero_grad()
    #         outputs, attns, dec_state = \
    #             self.model(mem_rtl, mem_rtl_length,
    #                        mem_ans, mem_ans_turn, mem_ans_length,
    #                        mem_que, mem_que_turn, mem_que_length,
    #                        cur_ans, cur_ans_length,
    #                        tgt, dec_state)
    #
    #         # 3. Compute loss in shards for memory efficiency.
    #         batch_stats = self.train_loss.sharded_compute_loss(
    #             batch, outputs, attns, j,
    #             trunc_size, self.shard_size, normalization)
    #         total_stats.update(batch_stats)
    #         report_stats.update(batch_stats)
    #
    #         # 4. Update the parameters and statistics.
    #         if self.grad_accum_count == 1:
    #             self.optim.step()
    #
    #         # If truncated, don't backprop fully.
    #         if dec_state is not None:
    #             dec_state.detach()



    def _gradient_accumulation(self, batch, normalization, total_stats,
                               report_stats):
        # dec_state = None
        # mem_rtl, mem_rtl_length = batch.src[0], batch.src[1]
        # mem_ans, mem_ans_turn, mem_ans_length = batch.memory_answer[0], batch.memory_answer[1], batch.memory_answer[2]
        # mem_que, mem_que_turn, mem_que_length = batch.memory_question[0], batch.memory_question[1], batch.memory_question[2]
        # cur_ans, cur_ans_length = batch.current_answer[0], batch.current_answer[1]
        # tgt = batch.tgt
        #
        # # 2. F-prop all but generator.
        # if self.grad_accum_count == 1:
        #     self.model.zero_grad()
        # outputs, attns, dec_state = \
        #     self.model(mem_rtl, mem_rtl_length,
        #                mem_ans, mem_ans_turn, mem_ans_length,
        #                mem_que, mem_que_turn, mem_que_length,
        #                cur_ans, cur_ans_length,
        #                tgt, dec_state)

        self._comb_src_qa_map(batch)

        self._coref_attn_target(batch)

        outputs, attns, dec_state = self._forward_prop(batch)

        # 3. Compute loss in shards for memory efficiency.
        batch_stats = self.train_loss.compute_loss(
            batch, outputs, attns, normalization)
        total_stats.update(batch_stats)
        report_stats.update(batch_stats)

        # 4. Update the parameters and statistics.
        if self.grad_accum_count == 1:
            self.optim.step()

        # If truncated, don't backprop fully.
        if dec_state is not None:
            dec_state.detach()

    def _forward_prop(self, batch):
        """forward propagation"""
        # 1, Get all data
        dec_state = None
        src = inputters.make_features(batch, 'src')
        qa = inputters.make_features(batch, 'qa')
        tgt = inputters.make_features(batch, 'tgt')
        _, src_lengths = batch.src
        _, qa_sent_lengths, qa_word_lengths = batch.qa
        # # make word features for qa
        # qa = qa.unsqueeze(-1)

        # 2. F-prop all but generator.
        if self.grad_accum_count == 1:
            self.model.zero_grad()
        outputs, attns, dec_state = \
            self.model(src, src_lengths,
                       qa, qa_sent_lengths, qa_word_lengths,
                       tgt, dec_state)
        return outputs, attns, dec_state


    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time


    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=False)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)
