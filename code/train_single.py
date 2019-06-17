#!/usr/bin/env python
"""
    Training on a single process
"""
from __future__ import division
from functools import reduce
import argparse
import os
import random
import torch
import torchtext

import onmt.opts as opts

from onmt.inputters.inputter import build_dataset_iter, lazily_load_dataset, \
    _load_fields, _collect_report_features
from onmt.model_builder import build_model
from onmt.utils.optimizers import build_optim
from onmt.trainer import build_trainer
from onmt.models import build_model_saver
from onmt.utils.logging import init_logger, logger


def _check_save_model_path(opt):
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' or 'generator' in name:
            dec += param.nelement()
    return n_params, enc, dec


def training_opt_postprocessing(opt, device_id):
    if opt.word_vec_size != -1:
        opt.src_word_vec_size = opt.word_vec_size
        opt.tgt_word_vec_size = opt.word_vec_size

    if opt.layers != -1:
        opt.enc_layers = opt.layers
        opt.dec_layers = opt.layers

    if opt.rnn_size != -1:
        opt.enc_rnn_size = opt.rnn_size
        opt.dec_rnn_size = opt.rnn_size
        if opt.model_type == 'text' and opt.enc_rnn_size != opt.dec_rnn_size:
            raise AssertionError("""We do not support different encoder and
                                 decoder rnn sizes for translation now.""")

    opt.brnn = (opt.encoder_type == "brnn")

    if opt.rnn_type == "SRU" and not opt.gpu_ranks:
        raise AssertionError("Using SRU requires -gpu_ranks set.")

    if torch.cuda.is_available() and not opt.gpuid:
        logger.info("WARNING: You have a CUDA device, \
                    should run with -gpu_ranks")

    if opt.seed > 0:
        torch.manual_seed(opt.seed)
        # this one is needed for torchtext random call (shuffled iterator)
        # in multi gpu it ensures datasets are read in the same order
        random.seed(opt.seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        if opt.seed > 0:
            # These ensure same initialization in multi gpu mode
            torch.cuda.manual_seed(opt.seed)

    return opt


def main(opt, device_id):
    opt = training_opt_postprocessing(opt, device_id)
    init_logger(opt.log_file)
    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        logger.info('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        model_opt = checkpoint['opt']
    else:
        checkpoint = None
        model_opt = opt

    # Peek the first dataset to determine the data_type.
    # (All datasets have the same data_type).
    first_dataset = next(lazily_load_dataset("train", opt))
    data_type = first_dataset.data_type  # concat, query, hier

    # Load fields generated from preprocess phase.
    fields = _load_fields(first_dataset, data_type, opt, checkpoint)

    # Report src/tgt features.
    src_features, tgt_features = _collect_report_features(fields)
    for j, feat in enumerate(src_features):
        logger.info(' * src feature %d size = %d'
                    % (j, len(fields[feat].vocab)))
    for j, feat in enumerate(tgt_features):
        logger.info(' * tgt feature %d size = %d'
                    % (j, len(fields[feat].vocab)))

    # Build model.
    model = build_model(model_opt, opt, fields, checkpoint)
    n_params, enc, dec = _tally_parameters(model)
    logger.info('encoder: %d' % enc)
    logger.info('decoder: %d' % dec)
    logger.info('* number of parameters: %d' % n_params)
    _check_save_model_path(opt)

    # Build optimizer.
    optim = build_optim(model, opt, checkpoint)

    # Build model saver
    model_saver = build_model_saver(model_opt, opt, model, fields, optim)

    trainer = build_trainer(opt, device_id, model, fields,
                            optim, data_type, model_saver=model_saver)

    def data_iter_fct(data_stage):
        """data_stage: train / valid"""
        pt_file = opt.data + '.' + data_stage + '.pt'
        logger.info('Loading {} dataset'.format(data_stage))
        dataset = torch.load(pt_file)
        logger.info('Loaded {} dataset'.format(data_stage))
        dataset.fields = fields
        is_train = True if data_stage=="train" else False
        batch_size = opt.batch_size if is_train else opt.valid_batch_size
        repeat = True if data_stage=="train" else False
        if opt.gpuid != -1:
            device = "cuda"
        else:
            device = "cpu"

        def sort_key(ex):
            """ Sort using length of source sentences. """
            return ex.total_tokens
        return torchtext.data.Iterator(dataset=dataset, batch_size=batch_size,
                                       device=device, train=is_train, sort=False,
                                       sort_key=sort_key, repeat=repeat)

    # Do training.
    trainer.train(data_iter_fct, opt.train_steps, opt.valid_steps)

    if opt.tensorboard:
        trainer.report_manager.tensorboard_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.add_md_help_argument(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)

    opt = parser.parse_args()
    main(opt, opt.gpuid)
