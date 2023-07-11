# TODO: Phase 1, build a GAN
import torch.nn as nn


# DCGAN: deep convolutional generative adversarial network
class Generator(nn.Module):
    def __init__(self, latent_size, ngf, out_channels):
        super(Generator, self).__init__()
        # define Generator layers
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_size, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

        self.initial_weights()

    def initial_weights(self):
        for module in self.modules():
            if isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            if isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight, mean=1.0, std=0.02)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        # forward pass
        return self.main(x)


# TODO: Discriminator

class Discriminator(nn.Module):
    def __init__(self, in_channels, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is ``(in_channels) x 64 x 64``
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

        # initialize the weights
        self.initial_weights()

    def initial_weights(self):
        # iterate over all modules
        for module in self.modules():
            # initialize weights for convolutional and linear layers with normal dist
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                # initial biases to zeros
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            # initialize weights for batch with normal dist
            if isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight, mean=1.0, std=0.02)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        # forward pass
        return self.main(x)


# binary cross entropy loss
criterion = nn.BCELoss()
