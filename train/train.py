# A script for handling the training process of the model.
import torch.optim as optim
from models.model import Generator, Discriminator
from utils.argparse_utils import opt

# Instantiate G and D
generator = Generator(100, 64, 3)
discriminator = Discriminator(3, 64)

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# Training loop

# Update D network first

# train with real data

# train with fake data

# Update G network

# train with real data

# train with fake data
