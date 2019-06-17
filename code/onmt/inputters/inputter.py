# -*- coding: utf-8 -*-
"""
    Defining general functions for inputters
"""
import glob
import os

from collections import Counter, defaultdict, OrderedDict
from itertools import count

import torch
import torchtext.data
import torchtext.vocab
from torchtext.data import NestedField

from onmt.inputters.dataset_base import UNK_WORD, PAD_WORD, BOS_WORD, EOS_WORD
from onmt.inputters.text_dataset import TextDataset
from onmt.utils.logging import logger
from onmt.utils.misc import load_pickle


def _getstate(self):
    return dict(self.__dict__, stoi=dict(self.stoi))


def _setstate(self, state):
    self.__dict__.update(state)
    self.stoi = defaultdict(lambda: 0, self.stoi)


torchtext.vocab.Vocab.__getstate__ = _getstate
torchtext.vocab.Vocab.__setstate__ = _setstate


def get_fields(n_src_features, n_qa_features, n_tgt_features, data_type):
    """
    Args:
        n_src_features: the number of source features to
            create `torchtext.data.Field` for.
        n_tgt_features: the number of target features to
            create `torchtext.data.Field` for.
        data_type: concat / query / hier
    Returns:
        A dictionary whose keys are strings and whose values are the
        corresponding Field objects.
    """
    return TextDataset.get_fields(n_src_features, n_qa_features, n_tgt_features, data_type)


def load_fields_from_vocab(vocab, data_type="text"):
    """
    Load Field objects from `vocab.pt` file.
    """
    vocab = dict(vocab)
    n_src_features = len(collect_features(vocab, 'src'))
    n_qa_features = len(collect_features(vocab, 'qa'))
    n_tgt_features = len(collect_features(vocab, 'tgt'))
    fields = get_fields(n_src_features, n_qa_features, n_tgt_features, data_type)
    for k, v in vocab.items():
        # Hack. Can't pickle defaultdict :(
        v.stoi = defaultdict(lambda: 0, v.stoi)
        fields[k].vocab = v
        if isinstance(fields[k], NestedField):
            fields[k].nesting_field.vocab = v
    return fields


def save_fields_to_vocab(fields):
    """
    Save Vocab objects in Field objects to `vocab.pt` file.
    """
    vocab = []
    for k, f in fields.items():
        if f is not None and 'vocab' in f.__dict__:
            f.vocab.stoi = f.vocab.stoi
            vocab.append((k, f.vocab))
    return vocab


def merge_vocabs(vocabs, vocab_size=None):
    """
    Merge individual vocabularies (assumed to be generated from disjoint
    documents) into a larger vocabulary.

    Args:
        vocabs: `torchtext.vocab.Vocab` vocabularies to be merged
        vocab_size: `int` the final vocabulary size. `None` for no limit.
    Return:
        `torchtext.vocab.Vocab`
    """
    merged = sum([vocab.freqs for vocab in vocabs], Counter())
    return torchtext.vocab.Vocab(merged,
                                 specials=[UNK_WORD, PAD_WORD,
                                           BOS_WORD, EOS_WORD],
                                 max_size=vocab_size)


def get_num_features(data_type, corpus_file, side):
    """
    Args:
        data_type (str): type of the source input.
            Options are [text|img|audio].
        corpus_file (str): file path to get the features.

    Returns:
        number of features on `side`.
    """
    # if data_type == 'concat':
    return TextDataset.get_num_features(corpus_file, side)
    # else:
    #     raise ValueError("Data type not implemented")


def make_features(batch, side, data_type='text'):
    """
    Args:
        batch (Tensor): a batch of source or target data.
        side (str): for source or for target.
        data_type (str): type of the source input.
            Options are [text|img|audio].
    Returns:
        A sequence of src/tgt tensors with optional feature tensors
        of size (len x batch).
    """
    if isinstance(batch.__dict__[side], tuple):
        data = batch.__dict__[side][0]
    else:
        data = batch.__dict__[side]

    feat_start = side + "_feat_"
    keys = sorted([k for k in batch.__dict__ if feat_start in k])
    features = [batch.__dict__[k] for k in keys]
    levels = [data] + features

    return torch.cat([level.unsqueeze(-1) for level in levels], -1)



def collect_features(fields, side="src"):
    """
    Collect features from Field object.
    """
    # assert side in ["src", "tgt"]
    feats = []
    for j in count():
        key = side + "_feat_" + str(j)
        if key not in fields:
            break
        feats.append(key)
    return feats


def collect_feature_vocabs(fields, side):
    """
    Collect feature Vocab objects from Field object.
    """
    feature_vocabs = []
    for j in count():
        key = side + "_feat_" + str(j)
        if key not in fields:
            break
        feature_vocabs.append(fields[key].vocab)
    return feature_vocabs


def build_dataset(fields, data_type=None, data_iter=None, data_path=None,
                  seq_length=0,
                  seq_length_trunc=0,
                  dynamic_dict=True, use_filter_pred=True):
    """
    Build src/tgt examples iterator from corpus files, also extract
    number of features.
    """
    # assert data_type is not None
    examples_iter, num_src_feats, num_qa_feats, num_tgt_feats = \
        TextDataset.make_text_examples_nfeats_tpl(
            data_iter, data_path, seq_length_trunc)

    dataset = TextDataset(fields, data_type, examples_iter,
                          num_src_feats, num_qa_feats, num_tgt_feats,
                          src_seq_length=seq_length,
                          dynamic_dict=dynamic_dict,
                          use_filter_pred=use_filter_pred)

    return dataset


def _build_field_vocab(field, counter, **kwargs):
    specials = list(OrderedDict.fromkeys(
        tok for tok in [field.unk_token, field.pad_token, field.init_token,
                        field.eos_token]
        if tok is not None))
    field.vocab = field.vocab_cls(counter, specials=specials, **kwargs)
    if isinstance(field, NestedField):
        field.nesting_field.vocab = field.vocab

def build_vocab(train_dataset, data_type, fields, share_vocab,
                src_vocab_path, src_vocab_size, src_words_min_frequency,
                tgt_vocab_path, tgt_vocab_size, tgt_words_min_frequency):
    """
    Args:
        data_type: concat / query / hier
        share_vocab(bool): share source and target vocabulary?
        src_vocab_path(string): Path to src vocabulary file.
        src_vocab_size(int): size of the source vocabulary.
        src_words_min_frequency(int): the minimum frequency needed to
                include a source word in the vocabulary.
        tgt_vocab_path(string): Path to tgt vocabulary file.
        tgt_vocab_size(int): size of the target vocabulary.
        tgt_words_min_frequency(int): the minimum frequency needed to
                include a target word in the vocabulary.

    Returns:
        Dict of Fields
    """
    counter = {}
    for k in fields:
        counter[k] = Counter()

    dataset = torch.load(train_dataset)
    logger.info(" * reloading %s." % train_dataset)

    for ex in dataset.examples:
        for k in fields:
            val = getattr(ex, k, None)
            if not fields[k].sequential:
                continue
            if isinstance(fields[k], torchtext.data.NestedField):
                val = [token for tokens in val for token in tokens]
            counter[k].update(val)

    _build_field_vocab(fields["tgt"], counter["tgt"],
                       max_size=tgt_vocab_size,
                       min_freq=tgt_words_min_frequency)
    logger.info(" * tgt vocab size: %d." % len(fields["tgt"].vocab))

    _build_field_vocab(fields["coref_tgt"], counter["coref_tgt"],
                       max_size=tgt_vocab_size,
                       min_freq=tgt_words_min_frequency)
    logger.info(" * coref tgt vocab size: %d." % len(fields["coref_tgt"].vocab))

    # here we only use src as the source
    # we will copy the vocabulary to others for sharing the source vocab
    _build_field_vocab(fields["src"], counter["src"],
                       max_size=src_vocab_size,
                       min_freq=src_words_min_frequency)
    logger.info(" * src vocab size: %d." % len(fields["src"].vocab))

    _build_field_vocab(fields["qa"], counter["qa"],
                       max_size=src_vocab_size,
                       min_freq=src_words_min_frequency)
    logger.info(" * qa vocab size: %d." % len(fields["qa"].vocab))

    # All datasets have same num of n_src_features,
    # getting the last one is OK.
    # # here we share counter between qa and src
    # _build_field_vocab(fields['qa_feat_0'], counter['qa_feat_0'] + counter['src_feat_0'])
    # _build_field_vocab(fields['src_feat_0'], counter['qa_feat_0'] + counter['src_feat_0'])
    # logger.info(" * 'qa_feat_0' vocab size: %d." % (len(fields['qa_feat_0'].vocab)))
    # logger.info(" * 'src_feat_0' vocab size: %d." % (len(fields['src_feat_0'].vocab)))

    for j in range(dataset.src_n_feats):
        key = "src_feat_" + str(j)
        _build_field_vocab(fields[key], counter[key])
        logger.info(" * %s vocab size: %d." %
                    (key, len(fields[key].vocab)))

    for j in range(dataset.qa_n_feats):
        key = "qa_feat_" + str(j)
        _build_field_vocab(fields[key], counter[key])
        logger.info(" * %s vocab size: %d." %
                    (key, len(fields[key].vocab)))

    for j in range(dataset.tgt_n_feats):
        key = "tgt_feat_" + str(j)
        _build_field_vocab(fields[key], counter[key])
        logger.info(" * %s vocab size: %d." %
                    (key, len(fields[key].vocab)))

    # Merge the input and output vocabularies.
    if share_vocab:
        # `tgt_vocab_size` is ignored when sharing vocabularies
        logger.info(" * merging src, tgt vocab...")
        merged_vocab = merge_vocabs(
            [fields["src"].vocab, fields["tgt"].vocab, fields["qa"].vocab],
            vocab_size=src_vocab_size)
        logger.info(" * merged vocab size: %d." % len(merged_vocab))
        fields["src"].vocab = merged_vocab
        # fields["src"].nesting_field.vocab = merged_vocab
        fields["tgt"].vocab = merged_vocab
        fields["qa"].vocab = merged_vocab
        fields["qa"].nesting_field.vocab = merged_vocab

    return fields


class OrderedIterator(torchtext.data.Iterator):
    """ Ordered Iterator Class """

    def create_batches(self):
        """ Create batches """
        if self.train:
            def _pool(data, random_shuffler):
                for p in torchtext.data.batch(data, self.batch_size * 100):
                    p_batch = torchtext.data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = _pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for b in torchtext.data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


class DatasetLazyIter(object):
    """ An Ordered Dataset Iterator, supporting multiple datasets,
        and lazy loading.

    Args:
        datsets (list): a list of datasets, which are lazily loaded.
        fields (dict): fields dict for the datasets.
        batch_size (int): batch size.
        batch_size_fn: custom batch process function.
        device: the GPU device.
        is_train (bool): train or valid?
    """

    def __init__(self, datasets, fields, batch_size, batch_size_fn,
                 device, is_train):
        self.datasets = datasets
        self.fields = fields
        self.batch_size = batch_size
        self.batch_size_fn = batch_size_fn
        self.device = device
        self.is_train = is_train

        self.cur_iter = self._next_dataset_iterator(datasets)
        # We have at least one dataset.
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)

    def __len__(self):
        # We return the len of cur_dataset, otherwise we need to load
        # all datasets to determine the real len, which loses the benefit
        # of lazy loading.
        assert self.cur_iter is not None
        return len(self.cur_iter)

    def _next_dataset_iterator(self, dataset_iter):
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, "cur_dataset"):
                self.cur_dataset.examples = None
                gc.collect()
                del self.cur_dataset
                gc.collect()

            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        # We clear `fields` when saving, restore when loading.
        self.cur_dataset.fields = self.fields

        # Sort batch by decreasing lengths of sentence required by pytorch.
        # sort=False means "Use dataset's sortkey instead of iterator's".
        return OrderedIterator(
            dataset=self.cur_dataset, batch_size=self.batch_size,
            batch_size_fn=self.batch_size_fn,
            device=self.device, train=self.is_train,
            sort=False, sort_within_batch=True,
            repeat=False)


def build_dataset_iter(datasets, fields, opt, is_train=True):
    """
    This returns user-defined train/validate data iterator for the trainer
    to iterate over. We implement simple ordered iterator strategy here,
    but more sophisticated strategy like curriculum learning is ok too.
    """
    batch_size = opt.batch_size if is_train else opt.valid_batch_size
    if is_train and opt.batch_type == "tokens":
        def batch_size_fn(new, count, sofar):
            """
            In token batching scheme, the number of sequences is limited
            such that the total number of src/tgt tokens (including padding)
            in a batch <= batch_size
            """
            # Maintains the longest src and tgt length in the current batch
            global max_src_in_batch, max_tgt_in_batch
            # Reset current longest length at a new batch (count=1)
            if count == 1:
                max_src_in_batch = 0
                max_tgt_in_batch = 0
            # Src: <bos> w1 ... wN <eos>
            max_src_in_batch = max(max_src_in_batch, len(new.src) + 2)
            # Tgt: w1 ... wN <eos>
            max_tgt_in_batch = max(max_tgt_in_batch, len(new.tgt) + 1)
            src_elements = count * max_src_in_batch
            tgt_elements = count * max_tgt_in_batch
            return max(src_elements, tgt_elements)
    else:
        batch_size_fn = None

    if opt.gpuid != -1:
        device = "cuda"
    else:
        device = "cpu"

    return DatasetLazyIter(datasets, fields, batch_size, batch_size_fn,
                           device, is_train)


def lazily_load_dataset(corpus_type, opt):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        return dataset

    # Only one inputters.*Dataset, simple!
    pt = opt.data + '.' + corpus_type + '.pt'
    yield _lazy_dataset_loader(pt, corpus_type)


def _load_fields(dataset, data_type, opt, checkpoint):
    if checkpoint is not None:
        logger.info('Loading vocab from checkpoint at %s.' % opt.train_from)
        fields = load_fields_from_vocab(
            checkpoint['vocab'], data_type)
    else:
        fields = load_fields_from_vocab(
            torch.load(opt.data + '.vocab.pt'), data_type)
    fields = dict([(k, f) for (k, f) in fields.items()
                   if k in dataset.examples[0].__dict__])
    # why I put this two lines here, vocab has been copied in function load_fields_from_vocab
    # fields['memory_question'].nesting_field.vocab = fields['tgt'].vocab
    # fields['memory_answer'].nesting_field.vocab = fields['memory_answer'].vocab

    logger.info(' * vocabulary size. source = %d; target = %d' %
                (len(fields['src'].vocab), len(fields['tgt'].vocab)))

    return fields


def _collect_report_features(fields):
    src_features = collect_features(fields, side='src')
    tgt_features = collect_features(fields, side='tgt')

    return src_features, tgt_features
