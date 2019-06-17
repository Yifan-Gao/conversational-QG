# -*- coding: utf-8 -*-

import torch
import pickle


def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def sequence_mask_herd(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    This function is specially designed for Hier Enc Dec
    Some sequences have all pad indices with length=0, we donot mask these
    because if we mask the whole sentence with -inf, softmax function will
    raise a error
    """
    batch_size = lengths.numel()
    lengths_ = lengths.clone()
    lengths_[lengths_ == 0] =  max_len + 1
    max_len = max_len or lengths.max()
    mask = (torch.arange(0, max_len)
            .type_as(lengths_)
            .repeat(batch_size, 1)
            .lt(lengths_.unsqueeze(1)))
    return mask


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def use_gpu(opt):
    """
    Creates a boolean if gpu used
    """
    return (hasattr(opt, 'gpu_ranks') and len(opt.gpu_ranks) > 0) or \
        (hasattr(opt, 'gpuid') and opt.gpuid > -1)


def load_pickle(loadpath, loadinfo):
    with open(loadpath, 'rb') as fh:
        print(loadinfo)
        dataset = pickle.load(fh)
        print('load pickle done')
    return dataset
