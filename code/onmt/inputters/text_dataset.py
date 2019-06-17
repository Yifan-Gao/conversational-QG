# -*- coding: utf-8 -*-
"""Define word-based embedders."""

from collections import Counter
from itertools import chain
import io
import codecs
import sys
import ujson as json

import torch
import torchtext

from onmt.inputters.dataset_base import (DatasetBase, UNK_WORD,
                                         PAD_WORD, BOS_WORD, EOS_WORD)
from onmt.utils.misc import aeq
from onmt.utils.logging import logger


class TextDataset(DatasetBase):
    """ Dataset for data_type=='text'

        Build `Example` objects, `Field` objects, and filter_pred function
        from text corpus.

        Args:
            fields (dict): a dictionary of `torchtext.data.Field`.
                Keys are like 'src', 'tgt', 'src_map', and 'alignment'.
            src_examples_iter (dict iter): preprocessed source example
                dictionary iterator.
            tgt_examples_iter (dict iter): preprocessed target example
                dictionary iterator.
            num_src_feats (int): number of source side features.
            num_tgt_feats (int): number of target side features.
            src_seq_length (int): maximum source sequence length.
            tgt_seq_length (int): maximum target sequence length.
            dynamic_dict (bool): create dynamic dictionaries?
            use_filter_pred (bool): use a custom filter predicate to filter
                out examples?
    """

    def __init__(self, fields, data_type, examples_iter,
                 num_src_feats=0, num_qa_feats=0, num_tgt_feats=0, src_seq_length=0,
                 dynamic_dict=True, use_filter_pred=True):
        self.data_type = data_type

        # self.src_vocabs: mutated in dynamic_dict, used in
        # collapse_copy_scores and in Translator.py
        self.src_vocabs = []

        self.src_n_feats = num_src_feats # num of src features
        self.qa_n_feats = num_qa_feats
        self.tgt_n_feats = num_tgt_feats

        if dynamic_dict:
            examples_iter = self._dynamic_dict(examples_iter)

        # Peek at the first to see which fields are used.
        ex, examples_iter = self._peek(examples_iter)

        # data_type == "concat"
        keys = ['id', 'filename', 'source', 'turn_id', 'total_tokens']
        for key in fields.keys():
            keys.append(key)
        # keys = ex.keys()

        out_fields = [(k, fields[k]) if k in fields else (k, None)
                      for k in keys]
        example_values = ([ex[k] for k in keys] for ex in examples_iter)

        # If out_examples is a generator, we need to save the filter_pred
        # function in serialization too, which would cause a problem when
        # `torch.save()`. Thus we materialize it as a list.
        out_examples = []
        for ex_values in example_values:
            example = self._construct_example_fromlist(
                ex_values, out_fields)
            out_examples.append(example)

        logger.info("{} Example before filter".format(len(out_examples)))

        def filter_pred(example):
            """ ? """
            # here we filter examples which exceeds the size
            return example.total_tokens <= src_seq_length

        filter_pred = filter_pred if use_filter_pred else lambda x: True

        super(TextDataset, self).__init__(
            out_examples, out_fields, filter_pred
        )

        logger.info("{} Example after filter".format(len(self.examples)))

    def sort_key(self, ex):
        """ Sort using length of source sentences. """
        # Default to a balanced sort, prioritizing tgt len match.
        # TODO: make this configurable.
        if hasattr(ex, "tgt"):
            return len(ex.src) + len(ex.tgt)
        return len(ex.src)

    @staticmethod
    def collapse_copy_scores(scores, batch, tgt_vocab, src_vocabs):
        """
        Given scores from an expanded dictionary
        corresponeding to a batch, sums together copies,
        with a dictionary word when it is ambigious.
        """
        offset = len(tgt_vocab)
        for b in range(batch.batch_size):
            blank = []
            fill = []
            index = batch.indices.data[b]
            src_vocab = src_vocabs[index]
            for i in range(1, len(src_vocab)):
                sw = src_vocab.itos[i]
                ti = tgt_vocab.stoi[sw]
                if ti != 0:
                    blank.append(offset + i)
                    fill.append(ti)
            if blank:
                blank = torch.Tensor(blank).type_as(batch.indices.data)
                fill = torch.Tensor(fill).type_as(batch.indices.data)
                scores[:, b].index_add_(1, fill,
                                        scores[:, b].index_select(1, blank))
                scores[:, b].index_fill_(1, blank, 1e-10)
        return scores

    @staticmethod
    def make_text_examples_nfeats_tpl(text_iter, text_path, truncate):
        """
        Args:
            text_iter(iterator): an iterator (or None) that we can loop over
                to read examples.
                It may be an openned file, a string list etc...
            text_path(str): path to file or None
            path (str): location of a src or tgt file.
            truncate (int): maximum sequence length (0 for unlimited).
            side (str): "src" or "tgt".

        Returns:
            (example_dict iterator, num_feats) tuple.
        """
        if text_iter is None:
            if text_path is not None:
                text_iter = TextDataset.make_text_iterator_from_file(text_path)
            else:
                return (None, 0)

        # All examples have same number of features, so we peek first one
        # to get the num_feats.
        examples_nfeats_iter = \
            TextDataset.make_examples(text_iter, truncate)

        first_ex = next(examples_nfeats_iter)
        _, num_src_feats, num_qa_feats, num_tgt_feats = first_ex

        # Chain back the first element - we only want to peek it.
        examples_nfeats_iter = chain([first_ex], examples_nfeats_iter)
        examples_iter = (ex for ex, _, _, _ in examples_nfeats_iter)

        return (examples_iter, num_src_feats, num_qa_feats, num_tgt_feats)

    @staticmethod
    def make_examples(text_iter, truncate):
        """
        Args:
            text_iter (iterator): iterator of text sequences
            truncate (int): maximum sequence length (0 for unlimited).
            side (str): "src" or "tgt".

        Yields:
            (word, features, nfeat) triples for each line.
        """
        for i, line in enumerate(text_iter):
            ex = json.loads(line)
            if truncate:
                for k, v in ex.items():
                    ex[k] = v[:truncate]


            src_words, src_feats, src_n_feats = \
                TextDataset.extract_text_features(ex['src'], 'src')
            qa_wordss, qa_featss, qa_n_feats = \
                TextDataset.extract_text_features(ex['qa'], 'qa')
            tgt_words, tgt_feats, tgt_n_feats = \
                TextDataset.extract_text_features(ex['tgt'], 'tgt')
            coref_tgt_words, coref_tgt_feats, coref_tgt_n_feats = \
                TextDataset.extract_text_features(ex['coref_tgt'], 'coref_tgt')

            example_dict = {
                'src': src_words,
                'qa': qa_wordss,
                'tgt': tgt_words,
                "indices": i,
                'id': ex['id'],
                'filename': ex['filename'],
                'source': ex['source'],
                'turn_id': ex['turn_id'],
                'total_tokens': ex['total_tokens'],
                'coref_tgt': coref_tgt_words,
                'coref_attn': ex['coref_attn'],
                'coref_score': ex['coref_score'],
                'sentence_label': ex['sentence_label'],
                'history_label': ex['history_label'],
            }

            if src_feats:
                prefix = 'src' + "_feat_"
                example_dict.update((prefix + str(j), f)
                                    for j, f in enumerate(src_feats))
            if qa_featss:
                prefix = 'qa' + "_feat_"
                example_dict.update((prefix + str(j), f)
                                    for j, f in enumerate(qa_featss))
            if tgt_feats:
                prefix = 'tgt' + "_feat_"
                example_dict.update((prefix + str(j), f)
                                    for j, f in enumerate(tgt_feats))

            yield example_dict, src_n_feats, qa_n_feats, tgt_n_feats

    @staticmethod
    def make_text_iterator_from_file(path):
        with codecs.open(path, "r", "utf-8") as corpus_file:
            for line in corpus_file:
                yield line

    @staticmethod
    def get_fields(n_src_features, n_qa_features, n_tgt_features, data_type):
        """
        Args:
            n_src_features (int): the number of source features to
                create `torchtext.data.Field` for.
            n_tgt_features (int): the number of target features to
                create `torchtext.data.Field` for.

        Returns:
            A dictionary whose keys are strings and whose values
            are the corresponding Field objects.
        """
        fields = {}

        fields["src"] = torchtext.data.Field(
            pad_token=PAD_WORD, include_lengths=True)

        fields["qa"] = torchtext.data.NestedField(
            torchtext.data.Field(pad_token=PAD_WORD), include_lengths=True)

        for j in range(n_src_features):
            fields["src_feat_" + str(j)] = \
                torchtext.data.Field(pad_token=PAD_WORD)

        for j in range(n_qa_features):
            fields["qa_feat_" + str(j)] = torchtext.data.NestedField(
                torchtext.data.Field(pad_token=PAD_WORD))

        fields["tgt"] = torchtext.data.Field(
            init_token=BOS_WORD, eos_token=EOS_WORD,
            pad_token=PAD_WORD)

        for j in range(n_tgt_features):
            fields["tgt_feat_" + str(j)] = \
                torchtext.data.Field(init_token=BOS_WORD, eos_token=EOS_WORD,
                                     pad_token=PAD_WORD)

        fields["coref_tgt"] = torchtext.data.Field(
            init_token=BOS_WORD, eos_token=EOS_WORD,
            pad_token=PAD_WORD)

        fields["coref_score"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.float,
            sequential=False)

        def make_coref_attn(data, vocab):
            # here only tensorlize the data
            coref_attn = torch.ones(len(data), 3) * -1
            for i, coref in enumerate(data):
                if coref is not None:
                    coref_attn[i, 0] = coref[0]
                    coref_attn[i, 1] = coref[1][0]
                    coref_attn[i, 2] = coref[1][1]
            return coref_attn

        fields['coref_attn'] = torchtext.data.Field(
            use_vocab=False, dtype=torch.long,
            postprocessing=make_coref_attn, sequential=False)


        # postprocess src to build dynamic vocab for copy
        def make_src(data, vocab):
            """ ? """
            src_size = max([t.size(0) for t in data])  # src longest sequence
            # src_vocab_size = max([t.max() for t in data]) + 1  # the vocab start from 0
            # alignment = torch.zeros(src_size, len(data), src_vocab_size)
            # for i, sent in enumerate(data):
            #     for j, t in enumerate(sent):
            #         alignment[j, i, t] = 1

            # here we only do simple tensorlize
            alignment = torch.ones(len(data), src_size) * -1
            # here -1 means PAD
            for i, sent in enumerate(data):
                for j, t in enumerate(sent):
                    alignment[i, j] = t
            return alignment

        # store src copy info
        fields["src_map"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.float,
            postprocessing=make_src, sequential=False)

        # postprocess src to build dynamic vocab for copy
        def make_qa(data, vocab):
            """ ? """
            sent_size = max([len(t) for t in data])  # len num. seq in ex
            word_size = max([len(s) for t in data for s in t])  # len seq
            # vocab_size = max([max(s) for t in data for s in t]) + 1  # the vocab start from 0
            # alignment = torch.zeros(sent_size * word_size, len(data), vocab_size)
            # for i, tokenss in enumerate(data):
            #     for j, tokens in enumerate(tokenss):
            #         start_jdx = word_size * j
            #         for k, t in enumerate(tokens):
            #             alignment[start_jdx + k, i, t] = 1

            alignment = torch.ones(len(data), sent_size, word_size) * -1
            for i, tokenss in enumerate(data):
                for j, tokens in enumerate(tokenss):
                    for k, t in enumerate(tokens):
                        alignment[i, j, k] = t
            return alignment

        # store qa copy info
        fields["qa_map"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.float,
            postprocessing=make_qa, sequential=False)

        # postprocess tgt to build dynamic vocab for copy
        def make_tgt(data, vocab):
            """ ? """
            tgt_size = max([t.size(0) for t in data])
            alignment = torch.zeros(tgt_size, len(data)).long()
            for i, sent in enumerate(data):
                alignment[:sent.size(0), i] = sent
            return alignment
        # store tgt copy info
        fields["alignment"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.long,
            postprocessing=make_tgt, sequential=False)

        fields["indices"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.long,
            sequential=False)

        def make_label(data, vocab):
            # transform label list to tensor, pad 0 if needed
            maxlen = max([len(t) for t in data])
            label = torch.zeros((len(data), maxlen), dtype=torch.long)
            for i, label_i in enumerate(data):
                label[i, :len(data[i])] = torch.Tensor(data[i]).type_as(label)
            return label

        fields["history_label"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.long,
            postprocessing=make_label, sequential=False)

        fields["sentence_label"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.long,
            postprocessing=make_label, sequential=False)

        return fields

    @staticmethod
    def get_num_features(corpus_file, side):
        """
        Peek one line and get number of features of it.
        (All lines must have same number of features).
        For text corpus, both sides are in text form, thus
        it works the same.

        Args:
            corpus_file (str): file path to get the features.
            side (str): 'src' or 'tgt'.

        Returns:
            number of features on `side`.
        """
        with codecs.open(corpus_file, "r", "utf-8") as cf:
            ex = json.loads(cf.readline())
            _, _, num_feats = TextDataset.extract_text_features(ex[side], side)
        return num_feats

    # Below are helper functions for intra-class use only.
    def _dynamic_dict(self, examples_iter):
        for example in examples_iter:
            src = example["src"]
            qa = example['qa']
            # src & qa share the same vocab
            src_vocab = torchtext.vocab.Vocab(Counter(list(src) + [token for tokens in qa for token in tokens]),
                                              specials=[UNK_WORD, PAD_WORD])
            self.src_vocabs.append(src_vocab)
            # Mapping source tokens to indices in the dynamic dict.
            src_map = torch.LongTensor([src_vocab.stoi[w] for w in src])
            example["src_map"] = src_map

            # Mapping source tokens to indices in the dynamic dict.
            qa_map = []
            for qa_i in qa:
                qa_map.append([src_vocab.stoi[w] for w in qa_i])
            example["qa_map"] = qa_map

            if "tgt" in example:
                tgt = example["tgt"]
                mask = torch.LongTensor(
                    [0] + [src_vocab.stoi[w] for w in tgt] + [0])
                example["alignment"] = mask
            yield example



