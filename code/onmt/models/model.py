""" memory network Model base class definition """
import torch.nn as nn


class PointerGenerator(nn.Module):
    """
    s2s+copy
    """
    def __init__(self, encoder, decoder):
        super(PointerGenerator, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, lengths, tgt, dec_state=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """
        tgt = tgt[:-1]  # exclude last target from inputs

        memory_bank, enc_final = self.encoder(src, lengths)
        enc_state = self.decoder.init_decoder_state(enc_final)
        decoder_outputs, dec_state, attns = self.decoder(tgt, memory_bank,
                                            enc_state, memory_lengths=lengths)

        return decoder_outputs, attns, dec_state


class QueryModel(nn.Module):
    """
    s2s+copy, use answer as query in decoder attention
    """
    def __init__(self, encoder, decoder):
        super(QueryModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, src_lengths, ans, ans_lengths, tgt, dec_state=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """
        tgt = tgt[:-1]  # exclude last target from inputs

        memory_bank, enc_final, ans_bank = self.encoder(src, src_lengths, ans,
                                                        ans_lengths)
        enc_state = self.decoder.init_decoder_state(enc_final)
        decoder_outputs, dec_state, attns = self.decoder(tgt, memory_bank,
                                enc_state, ans_bank, memory_lengths=src_lengths)

        return decoder_outputs, attns, dec_state


class MemModel(nn.Module):
    """
    memory augmented encoder-decoder
    """
    def __init__(self, encoder, decoder):
        super(MemModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, src_lengths,
                qa, qa_sent_lengths, qa_word_lengths,
                tgt, dec_state=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """
        tgt = tgt[:-1]  # exclude last target from inputs

        src_bank, qa_sent_bank, qa_word_bank, src_state = \
           self.encoder(src, src_lengths,
                        qa, qa_sent_lengths, qa_word_lengths)
        enc_state = self.decoder.init_decoder_state(src_state)
        decoder_outputs, dec_state, attns = \
            self.decoder(tgt,
                         src_bank, src_lengths,
                         qa_sent_bank, qa_sent_lengths,
                         qa_word_bank, qa_word_lengths,
                         enc_state)
        return decoder_outputs, attns, dec_state
