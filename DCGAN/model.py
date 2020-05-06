"""Defines a DCGAN Discriminator and Generator for synthesizing waste images."""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

####### Initialization (start) ########

# debug Flag
DEBUG=True

# decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# image size
image_size = 64

# number of color channels in image
num_channels = 3

# number of generator feature
ngf = 64

# number of descriminator feature
ndf = 64

# generator latent vector size
gvSize = 100

# number of generator latent vectors
ngv = 64

# batch of input latent vectors
fixed_noise = torch.randn(ngv, gvSize, 1, 1, device=device)

# labels
real_label = 1
fake_label = 0

####### Initialization (end) ########

####### Hyperparameter tuning ######

# batch-size for training
batch_size = 64

# learning-rate
learning_rate = 0.01

# adam optimizer
beta1 = 0.5

# number of epochs
num_epochs = 10

####### Hyperparameter tuning (end) ######

# DCGAN-Generator
class DCGAN_Generator(nn.Module):

  def __init__(self):
    super(DCGAN_Generator,self).__init__()

    # generator network
    self.generator = nn.Sequential(

        # layer-1 100(1x1) -> 512(4x4)
        nn.ConvTranspose2d(gvSize, (ngf*8), 4, 2, 0, bias=False),
        nn.BatchNorm2d(ngf*8),
        nn.ReLU(True),

        # layer-2  512x(4x4) -> 256x(8x8)
        nn.ConvTranspose2d((ngf*8), (ngf*4), 4, 2, 1,bias=False),
        nn.BatchNorm2d(ngf*4),
        nn.ReLU(True),

        # layer-3  256x(8x8) -> 128x(16x16)
        nn.ConvTranspose2d((ngf*4), (ngf*2), 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf*2),
        nn.ReLU(True),

        # layer-4  128x(16x16) -> 64x(32x32)
        nn.ConvTranspose2d((ngf*2), (ngf), 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf),
        nn.ReLU(True),

        # layer-4  64x(32x32) -> 3x(64x64)
        nn.ConvTranspose2d(ngf, num_channels, 4, 2, 1, bias=False),
        nn.Tanh()
    )

    # forward propagation
    def forward(self, input):
      return self.generator(input)



# DCGAN - Discriminator
class DCGAN_Discriminator(nn.Module):

  def __init__(self):
    super(DCGAN_Discriminator, self).__init__()

    # descriminator network
    self.discriminator = nn.Sequential(

      # layer-1 3x(64x64) -> 64x(32x32)
      nn.Conv2d(num_channels, ndf, 4, 2, 1, bias = False),
      nn.LeakyReLU(0.2, True),

      # layer-2 64x(32x32) -> 128x(16x16)
      nn.Conv2d(ndf, (ndf*2), 4, 2, 1, bias = False),
      nn.BatchNorm2d(ndf*2),
      nn.LeakyReLU(0.2, True),

      # layer-3 128x(16x16) -> 256x(8x8)
      nn.Conv2d((ndf*2), (ndf*4), 4, 2, 1, bias = False),
      nn.BatchNorm2d(ndf*4),
      nn.LeakyReLU(0.2, True),

      # layer-4 256x(8x8) -> 512x(4x4)
      nn.Conv2d((ndf*4), (ndf*8), 4, 2, 1, bias = False),
      nn.BatchNorm2d(512),
      nn.LeakyReLU(0.2, True),

      # layer-5 512x(4x4) -> 1x(1x1)
      nn.Conv2d((ndf*8), 1, 4, 1, 0, bias = False),
      nn.Sigmoid(),

  )

  def forward(self, input):
    return self.discriminator(self, input)


def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    nn.init.normal_(m.weight.data, 0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
    nn.init.normal_(m.weight.data, 1.0, 0.02)
    nn.init.constant_(m.bias.data, 0)


netG = DCGAN_Generator().to(device)
netG.apply(weights_init)

netD = DCGAN_Discriminator().to(device)
netD.apply(weights_init)

if DEBUG == True:
  # print Generator network
  print(netG)
  # print Descriminator network
  print(netD)

