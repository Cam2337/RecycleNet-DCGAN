#!/usr/bin/env python3
"""Training module for DCGAN."""

import argparse
import logging
logging.root.setLevel(logging.INFO)
import os
from typing import Any, Dict, List, Tuple

import model
import utils
import math
import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from torchvision.utils import save_image

# Constants #

RESULTS_DIR = 'results'
NUM_CHANNELS = 3
OPTIMAL_D_SCORE = 0.5

FAKE_LABEL = 0
REAL_LABEL = 1
SOFT_COEFF = 0.25

MIN_LR = 10e-5
MAX_LR = 1.0

CHECKPOINT_FREQ = 500
IMG_SAVE_COEF = 0.98
GAN_ERROR_THRESHOLD = 0.98
GRID_SIZE = 64
LOGGING_FREQ = 10
NUM_FAKES = 500

# Create figures directory
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
MODEL_DIR = os.path.join(RESULTS_DIR, 'model')

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

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
        'name',
        help='The base name of this batch of synthesized images, e.g. "metal".',
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
        '--netD-checkpoint',
        help='Initializes the Discriminator from the specified checkpoint.',
        type=str,
    )
    parser.add_argument(
        '--netG-checkpoint',
        help='Initializes the Generator from the specified checkpoint.',
        type=str,
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
        '--num-trials',
        help='The number of trials to use during hyperparameter searching.',
        type=int,
        default=10,
    )
    parser.add_argument(
        '--num-trial-cpus',
        help='The number of CPUs available during hyperparameter searching.',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--num-trial-gpus',
        help='The number of GPUs available during hyperparameter searching.',
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

def synthesize_training_data(
    netG: model.Generator, fixed_noise: torch.Tensor, epoch: int):
    """Saves synthesized images given a latent vector and a generator model."""
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
        for s in range(NUM_FAKES):
            save_image(
                fake[s,:,:,:],
                os.path.join(FIGURES_DIR, f'fake_out_{epoch}_{s}.png'))

def load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    filepath: str) -> Tuple[int, float]:
    """Loads model and optimizer state from the provided .model file."""
    if not os.path.exists(filepath):
        raise ValueError(f'Filepath: {filepath} does not exist!')

    logging.info(f'Loading checkpoint: {filepath}...')
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return (checkpoint['epoch'], checkpoint['loss'])

def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str):
    """Saves model and optimizer state to a filepath."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)

def train(config: Dict[str, Any]) -> Tuple[List[float], List[float], List[torch.Tensor]]:
    """The primary function for DCGAN training.

    Note: Per GANHacks, Discriminator training is conducted in *two* separate
    batch training sessions: one with all-real data, and one with all-fake data.
    See more at: https://github.com/soumith/ganhacks.forward

    Args:
        config: A dict with the following parameters:
            * netG: The Generator to train.
            * netG_checkpoint: An optional .model checkpoint to load from.
            * netD: The Discriminator to train.
            * netD_checkpoint: An optional .model checkpoint to load from.
            * dataloader: The PyTorch DataLoader used to iterate through data.
            * device: The device that the models are loaded onto.
            * learning_rate: The learning rate to apply during updates.
            * num_epochs: The number of training epochs.
            * beta1: The Beta1 parameter of Adam optimization.
            * beta2: The Beta2 parameter of Adam optimization.
            * pre_epoch: An optional hook for processing prior-to the epoch.
            * post_epoch: An optional hook for processing post-epoch.
    Returns:
        A tuple of lists containing the loss of the Generator and the
        Discriminator, respectively, from each training iteration, along with
        a list of images.
    """

    # Set parameters
    netG = config['netG']
    netD = config['netD']
    dataloader = config['dataloader']
    device = config['device']
    learning_rate = config['learning_rate']
    num_epochs = config['num_epochs']
    beta1 = config['beta1']
    beta2 = config['beta2']

    # Retrieve optional configuration parameters
    pre_epoch = config.get('pre_epoch')
    post_epoch = config.get('post_epoch')
    netG_checkpoint = config.get('netG_checkpoint')
    netD_checkpoint = config.get('netD_checkpoint')

    # Batch of input latent vectors
    fixed_noise = torch.randn(
        NUM_FAKES, netG.latent_vector_size, 1, 1, device=device)

    # Setup loss function and optimizers
    lossF = nn.BCELoss()
    optD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(beta1, beta2))
    optG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(beta1, beta2))

    # Load from saved state, if provided
    checkpoint_epochs = []
    if netG_checkpoint is not None:
        G_epoch, _ = load_checkpoint(netG, optG, netG_checkpoint)
        checkpoint_epochs.append(G_epoch)
    if netD_checkpoint is not None:
        D_epoch, _ = load_checkpoint(netD, optD, netD_checkpoint)
        checkpoint_epochs.append(D_epoch)

    # Dump model configuration
    logging.info(f'Generator:\n{netG}')
    logging.info(f'Discriminator:\n{netD}')

    # Main training loop
    img_list = []
    G_losses = []
    D_losses = []
    D_batch_scores = []
    iters = 0

    logging.info('Starting training...')
    epoch = min(checkpoint_epochs) if checkpoint_epochs else 0
    while epoch < num_epochs:

        logging.info(f'Starting epoch: {epoch}...')

        # Call into pre-epoch handler, if present
        if pre_epoch is not None:
            pre_epoch(
                epoch=epoch,
                G_losses=G_losses,
                D_losses=D_losses,
                D_batch_scores=D_batch_scores)

        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Real data
            netD.zero_grad()

            ## Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), REAL_LABEL, device=device)
            utils.add_label_noise(label, p_flip=0.05)

            r_label_soft = (
                REAL_LABEL +
                (torch.randn((b_size,), device=device)*SOFT_COEFF))
            r_label_noisy_soft = torch.mul(label, r_label_soft)

            ## Forward pass real data through discriminator
            output = netD(real_cpu).view(-1)

            ## Calculate loss on all-real batch; calculate gradients
            errD_real = lossF(output, r_label_noisy_soft)
            errD_real.backward()
            D_x = output.mean().item()

            ## Fake data
            noise = torch.randn(
                b_size, netG.latent_vector_size, 1, 1, device=device)

            ## Generate fake image batch with G
            fake = netG(noise)
            label.fill_(FAKE_LABEL)
            utils.add_label_noise(label, p_flip=0.05)
            f_label_noisy_soft = (
                label +
                torch.abs(torch.randn((b_size,), device=device))*SOFT_COEFF)

            ## Classify all fake batch with D
            output = netD(fake.detach()).view(-1)

            ## Calculate D's loss on the all-fake batch
            errD_fake = lossF(output, f_label_noisy_soft)

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
            label.fill_(REAL_LABEL)  # fake labels are real for generator cost

            # Since we just updated D, perform another forward pass of all-fake
            # batch through D
            output = netD(fake).view(-1)

            # Calculate G's loss based on this output
            errG = lossF(output, label)

            # Calculate gradients for G; Update
            errG.backward()
            D_G_z2 = output.mean().item()
            optG.step()

            # Save losses for plotting
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Save discriminator output
            D_batch_scores.append(D_x)

            # Output training stats
            if i % LOGGING_FREQ == 0:
                logging.info(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}]\t'
                            f'Loss_D: {errD.item():.4f}\tLoss_G: '
                            f'{errG.item():.4f}\t'
                            f'D(x): {D_x:.4f}\tD(G(z)): '
                            f'{D_G_z1:.4f} / {D_G_z2:.4f}')


            # Save checkpoint; dump model and optimizer states along with grid
            if ((iters % CHECKPOINT_FREQ == 0) or
                ((epoch == num_epochs - 1) and (i == len(dataloader) - 1))):
                img_list.append(
                    vutils.make_grid(
                        fake[0:GRID_SIZE], padding=2, normalize=True))
                save_checkpoint(
                    netG, optG, epoch, errG.item(),
                    os.path.join(MODEL_DIR, f'modelG_{epoch}.model'))
                save_checkpoint(
                    netD, optD, epoch, errD.item(),
                    os.path.join(MODEL_DIR, f'modelD_{epoch}.model'))

            # If we're sufficiently late into training, and the generator is having
            # success fooling the discriminator, synthesize training images
            if ((epoch >= math.floor(IMG_SAVE_COEF * num_epochs)) and
                (errG.item() <= GAN_ERROR_THRESHOLD)):
                synthesize_training_data(fixed_noise, epoch)

            iters += 1

        # Call into post-epoch handler, if present
        epoch += 1
        if post_epoch is not None:
            post_epoch(
                epoch=epoch,
                G_losses=G_losses,
                D_losses=D_losses,
                avg_D_batch_scores=D_batch_scores)

    return (G_losses, D_losses, img_list)

def main():
    """main."""
    args = parse_args()

    device = torch.device((
        'cuda:0' if torch.cuda.is_available and args.num_gpus > 0 else 'cpu'))
    logging.info(f'Running with device: {device}')

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

    # Load dataset and resize
    dataset = utils.data_synthesis(
        os.path.abspath(args.dataroot),
        image_size=(args.image_size, args.image_size, NUM_CHANNELS),
        custom_transforms=[
            transforms.ColorJitter(
                brightness=0.05,
                contrast=0.05,
                saturation=0.05,
                hue=0.03,
            ),
            transforms.RandomCrop(size=args.image_size),
            transforms.RandomHorizontalFlip(p=0.9),
            transforms.RandomVerticalFlip(p=0.9),
            transforms.Lambda(lambd=lambda img: img) # Identity transform
        ]
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    config = {
        'netG': netG,
        'netG_checkpoint': args.netG_checkpoint,
        'netD': netD,
        'netD_checkpoint': args.netD_checkpoint,
        'dataloader': dataloader,
        'device': device,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'beta1': args.beta1,
        'beta2': args.beta2,
    }

    logging.info('Beginning training loop...')
    G_losses, D_losses, img_list = train(config)
    utils.plot_results(
        device=device,
        dataloader=dataloader,
        G_losses=G_losses,
        D_losses=D_losses,
        img_list=img_list,
        name=args.name,
        outdir=FIGURES_DIR,
    )

if __name__ == '__main__':
    main()
