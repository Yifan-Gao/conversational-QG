"""Module defining encoders."""
from onmt.encoders.encoder import EncoderBase
from onmt.encoders.rnn_encoder import RNNEncoder
from onmt.encoders.mean_encoder import MeanEncoder

__all__ = ["EncoderBase", "RNNEncoder",
           "MeanEncoder", "PermutationWrapper", "PermutationWrapper2D"]
