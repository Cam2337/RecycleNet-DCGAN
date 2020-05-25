"""A collection of utilities for DCGAN training, testing, and validation."""

import os
from typing import Any, List, Tuple

import constants

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from numpy.random import choice

# Constants #

CHANNEL_MEAN = 0.5
CHANNEL_STDEV = 0.5

# Public Functions #

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

def data_synthesis(
    root: str,
    image_size: Tuple[int, int, int],
    custom_transforms: List[constants.Transforms]) -> torch.utils.data.ConcatDataset:
    """Augments the dataset at `root` with the supplied transforms.

    Each supplied transform will denote a _new_ copy of the dataset.
    All supplied transforms are applied _after_ performing a standard
    transformation pipeline of:
        1. Resize (to the supplied `image_size`)
        2. Perform center cropping (to the supplied `image_size`)

    Once the custom transformation has been done, the following additional
    transforms are made to standardize the data:
        1. Coercion from a PIL image or numpy.ndarray (H x W x C) in the range
           [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range
           [0.0, 1.0]
        2. Normalization of each channel around mean of 0.5 and stddev of 0.5.

    If you wish to have *multiple* transforms apply in a single dataset
    generation step, supply a transforms.Compose instance.

    Args:
        root: The path to the root directory of images to augment.
        image_size: A tuple describing (H, W, C) for an image.
        transforms: A list of torchvision.transforms.Transforms to compose
        together.

    Returns:
        An instance of `torch.utils.data.ConcatDataset`.

    Raises:
        ValueError: If the path specified by `root` does not exist.
    """
    if not os.path.exists(root):
        raise ValueError(f'Path: {root} does not exist!')

    datasets = []
    num_channels = image_size[2]
    for custom_transform in custom_transforms:
        dataset = dset.ImageFolder(
            root=root,
            transform=transforms.Compose([
                transforms.Resize(image_size[:-1]), # Expects (H, W)
                transforms.CenterCrop(image_size[:-1]), # Expects (H, W)

                custom_transform,

                transforms.ToTensor(),
                transforms.Normalize(
                    (CHANNEL_MEAN,) * num_channels, # Mean
                    (CHANNEL_STDEV,) * num_channels, # StdDev
                ),
            ])
        )
        datasets.append(dataset)
    return torch.utils.data.ConcatDataset(datasets)

def add_label_noise(y: torch.Tensor, p_flip: float):
    """Flips binary labels with some probability `p_flip`."""
    n_select = int(p_flip * y.shape[0])
    flip_ix = choice([i for i in range(y.shape[0])], size=n_select)
    y[flip_ix] = 1 - y[flip_ix]

def plot_results(**kwargs: Any):
    """Plots a batch of real and fake images along with loss metrics.

    Args:
        **kwargs: A key/value mapping of the following:
            * device: The torch.device to run render with.
            * dataloader: The torch.utils.data.DataLoader used.
            * G_losses: A list of Generator losses from each iteration.
            * D_losses: A list of Discriminator losses from each iteration.
            * img_list: A list of fake images.
            * name: A string name for the resulting plots.
            * outdir: The directory to save images to.
    """
    # Initialize parameters
    device = kwargs['device']
    dataloader = kwargs['dataloader']
    G_losses = kwargs['G_losses']
    D_losses = kwargs['D_losses']
    img_list = kwargs['img_list']
    name = kwargs['name']
    outdir = kwargs['outdir']

    plt.figure(figsize=(10,5))
    plt.title('Generator and Discriminator Loss During Training')
    plt.plot(G_losses,label='G')
    plt.plot(D_losses,label='D')
    plt.xlabel('iterations')
    plt.ylabel('Loss')
    plt.legend()

    # Save loss graph
    plt.savefig(os.path.join(outdir, f'{name}_gd_loss.png'), format='png')

    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis('off')
    plt.title('Real Images')
    plt.imshow(
        np.transpose(
            vutils.make_grid(
                real_batch[0].to(device)[:64],
                padding=5,
                normalize=True
            ).cpu(),
            (1,2,0),
        ),
    )

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis('off')
    plt.title('Fake Images')
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))

    # Save real/fake comparison
    plt.savefig(
        os.path.join(outdir, f'{name}_real_fake_comparison.png'), format='png')
