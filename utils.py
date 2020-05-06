"""A collection of utilities for DCGAN training, testing, and validation."""

import torch.nn as nn

def dcgan_weights_reinit(model: nn.Module):
    """Takes an initialized model and re-initializes weights.

    The DCGAN paper specifies that all model weights shall be randomly
    initialized to a Normal distribution with mean=0, stddev=0.02.

    See more: https://arxiv.org/pdf/1511.06434.pdf.

    Args:
        model: The initialized model whose weights to re-initialize.
    """
    # TODO(ctew): Find out why BatchNorm mean=0 and we initialize bias?
    # TODO(ctew): Parameterize weights, layer names, etc?
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)
