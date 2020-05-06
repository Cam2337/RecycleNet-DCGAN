#!/usr/bin/env python3

"""Defines a DCGAN Discriminator and Generator for synthesizing waste images."""
import os
from typing import Any, Dict

import numpy as np
import torch.nn as nn

import torch
import torchvision.transforms as transforms
import torch.optim as optim

# Classes #

class Generator(nn.Module):
    """Maps a latent space vector (z) to data-space."""

    def __init__(self, **kwargs: Dict[str, Any]):
        """Creates a new instance of Generator.

        Args:
            **kwargs: An optional set of key/value pair arguments:
                * latent_vector_size: The size of the Normally-distributed
                latent-vector input. Defaults to 100.
                * num_features: The feature size of the output data. Defaults
                to 64 (e.g. 64x64 images).
                * num_channels: The number of channels of the output data.
                Defaults to 3 (e.g. RGB images).
        """

        super(Generator, self).__init__()

        latent_vecor_size = kwargs.get('latent_vector_size', 100)
        num_features = kwargs.get('num_features', 64)
        num_channels = kwargs.get('num_channels', 3)


        self.generator = nn.Sequential(
            # layer-1 100(1x1) -> 512(4x4)
            nn.ConvTranspose2d(
                in_channels=latent_vecor_size,
                out_channels=(num_features * 8),
                kernel_size=4,
                stride=2,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=num_features * 8),
            nn.ReLU(inplace=True),

            # layer-2  512x(4x4) -> 256x(8x8)
            nn.ConvTranspose2d(
                in_channels=(num_features * 8),
                out_channels=(num_features * 4),
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_feautres=num_features * 4),
            nn.ReLU(inplace=True),

            # layer-3  256x(8x8) -> 128x(16x16)
            nn.ConvTranspose2d(
                in_channels=(num_features * 4),
                out_channels=(num_features * 2),
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=num_features * 2),
            nn.ReLU(inplace=True),

            # layer-4  128x(16x16) -> 64x(32x32)
            nn.ConvTranspose2d(
                in_channels=(num_features * 2),
                out_channels=(num_features),
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=num_features),
            nn.ReLU(inplace=True),

            # layer-5  64x(32x32) -> 3x(64x64)
            nn.ConvTranspose2d(
                in_channels=num_features,
                out_channels=num_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.Tanh()
        )

    # forward propagation
    def forward(self, input):
        return self.generator(input)

class Discriminator(nn.Module):
    """
    A binary classification network that takes an image as input and outputs
    a scalar probability that the input image is real (as opposed to fake).
    """

    def __init__(self, **kwargs: Dict[str, Any]):
        """Creates a new instance of Discriminator.

        Args:
            **kwargs: An optional set of key/value pair arguments:
                * num_features: The feature size of the input data. Defaults
                to 64 (e.g. 64x64 images).
                * num_channels: The number of channels of the input data.
                Defaults to 3 (e.g. RGB images).
        """
        super(Discriminator, self).__init__()

        num_features = kwargs.get('num_features', 64)
        num_channels = kwargs.get('num_channels', 3)

        # descriminator network
        self.discriminator = nn.Sequential(
            # layer-1 3x(64x64) -> 64x(32x32)
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_features,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # layer-2 64x(32x32) -> 128x(16x16)
            nn.Conv2d(
                in_channels=num_features,
                out_channels=(num_features * 2),
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=num_features * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # layer-3 128x(16x16) -> 256x(8x8)
            nn.Conv2d(
                in_channels=(num_features * 2),
                out_channels=(num_features * 4),
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=num_features * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # layer-4 256x(8x8) -> 512x(4x4)
            nn.Conv2d(
                in_channels=(num_features * 4),
                out_channels=(num_features * 8),
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            #  layer-5 512x(4x4) -> 1x(1x1)
            nn.Conv2d(
                in_channels=(num_features * 8),
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.discriminator(input)
