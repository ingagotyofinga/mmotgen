# A script for handling the training process of the model.
import torch.optim as optim
from models.model import Generator, Discriminator, criterion
from utils.argparse_utils import opt
import random

# Instantiate G and D
generator = Generator(100, 64, 3)
discriminator = Discriminator(3, 64)

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# Training loop

# TODO: Update D network first

# TODO: train with real data

# TODO: train with fake data

# TODO: Update G network

# TODO: train with real data

# TODO: train with fake data
