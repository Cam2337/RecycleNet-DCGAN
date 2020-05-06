#!/usr/bin/env python3
"""Training module for DCGAN."""

import argparse
import logging
logging.root.setLevel(logging.INFO)
import os
from typing import List, Tuple

import model
import utils

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

# Public Functions #

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataroot',
        help='The root of the directory whose image data to process.',
        type=str,
    )
    parser.add_argument(
        '--batch-size',
        help='The batch size for batch training.',
        type=int,
        default=128,
    )
    parser.add_argument(
        '--beta1',
        help='The Beta1 parameter for Adam Optimization.',
        type=float,
        default=0.5,
    )
    parser.add_argument(
        '--beta2',
        help='The Beta2 parameter for Adam Optimization.',
        type=float,
        default=0.999,
    )
    parser.add_argument(
        '--image-size',
        help='The size of the images.',
        type=int,
        default=64,
    )
    parser.add_argument(
        '--learning-rate',
        help='The learning rate to apply during parameter updates.',
        type=float,
        default=0.0002,
    )
    parser.add_argument(
        '--num-epochs',
        help='The number of training epochs to run.',
        type=int,
        default=5,
    )
    parser.add_argument(
        '--num-gpus',
        help='The number of GPUs available for training. Use 0 for CPU.',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--num-workers',
        help='The number of parallel workers for the DataLoader.',
        type=int,
        default=2,
    )

    # Perform some basic argument validation
    args = parser.parse_args()
    if not os.path.exists(args.dataroot) and os.path.isdir(args.dataroot):
        raise ValueError(f'{args.dataroot} is not a valid directory.')
    return args

def train(
    netG: model.Generator,
    netD: model.Discriminator,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    learning_rate: float,
    num_epochs: int,
    beta1: int,
    beta2: int) -> Tuple[List[float], List[float]]:
    """The primary function for DCGAN training.

    Note: Per GANHacks, Discriminator training is conducted in *two* separate
    batch training sessions: one with all-real data, and one with all-fake data.
    See more at: https://github.com/soumith/ganhacks.forward

    Args:
        netG: The Generator to train.
        netD: The Discriminator to train.
        dataloader: The PyTorch DataLoader used to iterate through image data.
        device: The device that the models are loaded onto.
        learning_rate: The learning reat to apply during parameter updates.
        num_epochs: The number of training epochs.
        beta1: The Beta1 parameter of Adam optimization.
        beta2: The Beta2 parameter of Adam optimization.

    Returns:
        A tuple of lists containing the loss of the Generator and the
        Discriminator, respectively, from each training iteration.
    """

    # Batch of input latent vectors
    fixed_noise = torch.randn(
        netG.num_features, netG.latent_vector_size, 1, 1, device=device)

    # Setup loss function and optimizers
    lossF = nn.BCELoss()
    optD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(beta1, beta2))
    optG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(beta1, beta2))

    # Main training loop
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    # Labels
    real_label = 1
    fake_label = 0

    logging.info('Starting training...')
    for epoch in range(num_epochs):
        logging.info(f'Starting epoch: {epoch}...')
        for i, data in enumerate(dataloader):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Real data
            netD.zero_grad()

            ## Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)

            ## Forward pass real data through discriminator
            output = netD(real_cpu).view(-1)

            ## Calculate loss on all-real batch; calculate gradients
            errD_real = lossF(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            ## Fake data
            noise = torch.randn(
                b_size, netG.latent_vector_size, 1, 1, device=device)

            ## Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)

            ## Classify all fake batch with D
            output = netD(fake.detach()).view(-1)

            ## Calculate D's loss on the all-fake batch
            errD_fake = lossF(output, label)

            ## Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            ## Add the gradients from the all-real and all-fake batches; Update
            errD = errD_real + errD_fake
            optD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost

            # Since we just updated D, perform another forward pass of all-fake
            # batch through D
            output = netD(fake).view(-1)

            # Calculate G's loss based on this output
            errG = lossF(output, label)

            # Calculate gradients for G; Update
            errG.backward()
            D_G_z2 = output.mean().item()
            optG.step()

            # Output training stats
            if i % 50 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}]\t'
                      f'Loss_D: {errD.item():.4f}\tLoss_G: {errG.item():.4f}\t'
                      f'D(x): {D_x:.4f}\tD(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')

            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    return (G_losses, D_losses)

def plot_results(
    dataloader: torch.utils.data.DataLoader,
    G_losses: List[float],
    D_losses: List[float]):
    """Plots a batch of real and fake images from the last epoch."""
    plt.figure(figsize=(10,5))
    plt.title('Generator and Discriminator Loss During Training')
    plt.plot(G_losses,label='G')
    plt.plot(D_losses,label='D')
    plt.xlabel('iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

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
                real_batch[0].to(model.device)[:64],
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
    plt.show()

def main():
    """main."""
    args = parse_args()

    device = torch.device((
        'cuda:0' if torch.cuda.is_available and args.num_gpus > 0 else 'cpu'))

    # Initialize models
    netG = model.Generator().to(device)
    netD = model.Discriminator().to(device)
    if device.type == 'cuda' and args.num_gpus > 1:
        netG = nn.DataParallel(netG, list(range(args.num_gpus)))
        netD = nn.DataParallel(netD, list(range(args.num_gpus)))

    # Apply DCGAN paper weight-reinitialization
    # See more: https://arxiv.org/pdf/1511.06434.pdf
    netG.apply(utils.dcgan_weights_reinit)
    netD.apply(utils.dcgan_weights_reinit)

    logging.info(f'Generator:\n{netG}')
    logging.info(f'Discriminator:\n{netD}')

    # Load dataset and resize
    dataset = dset.ImageFolder(
        root=args.dataroot,
        transform=transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)
            ),
        ]),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # Run training and plot results
    G_losses, D_losses = train(
        netG,
        netD,
        dataloader,
        device,
        args.learning_rate,
        args.num_epochs,
        args.beta1,
        args.beta2,
    )
    plot_results(dataloader, G_losses, D_losses)

if __name__ == '__main__':
    main()
