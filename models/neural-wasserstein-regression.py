import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import geomloss
import matplotlib.pyplot as plt
import time
import math
import random
from sklearn.model_selection import train_test_split
from data.simulate_data import DataSimulator
import pandas as pd

start_time = time.time()
torch.manual_seed(42)

class EnhancedOTMapNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EnhancedOTMapNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def box_kernel(mu_0, mu_i, bandwidth):
    """
    Computes indicator kernel for custom_loss

    Parameters:
    mu_0 (torch tensor): tensor of samples from reference measure
    mu_2 (torch tensor): tensor of samples from source measure
    bandwidth (float): kernel bandwidth

    Returns:
    kernel evaluated at reference and source measure
    """

    sink_loss = geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=0.05)
    # compute wasserstein distance between source and reference measure
    wasserstein_distance = sink_loss(mu_0, mu_i)
    # indicator with bandwidth as threshold
    indicator = torch.tensor(1.0) if wasserstein_distance <= bandwidth else torch.tensor(0.0)
    # kernel is scaled indicator
    kernel = (1 / (2 * bandwidth)) * indicator
    return kernel

def custom_loss(push, localdf, source, target, bw, blur=0.05, p=2, batch_size=1):
    """
    Computes loss between predicted and ground truth

    Parameters:
    push (torch tensor): predicted local pushfoward map centered
                        about localdf and evaluated at source
    localdf (torch tensor): tensor of samples from reference measure
    source (torch tensor): tensor of samples from source measure
    target (torch tensor): tensor of samples from target measure
    bw (float): kernel bandwidth
    blur (float): sinkhorn divergence parameter (epsilon)
    p (int): sinkhorn divergence parameter (Wp distance)
    batch_size (int): batch size for loss computation

    Returns:
    loss (torch sum): loss between predicted and ground truth
    """

    sink_loss = geomloss.SamplesLoss(loss='sinkhorn', p=p, blur=blur)
    wasserstein_distance = [sink_loss(push[i], target[i]) for i in range(batch_size)]
    wasserstein_distance = torch.stack(wasserstein_distance)
    kernel = [box_kernel(localdf, source[i], bw) for i in range(batch_size)]
    kernel = torch.stack(kernel)
    total_loss = torch.sum(wasserstein_distance * kernel)
    return total_loss

# SIMULATE DATA
num_distributions = 50
num_samples = 100
num_dimensions = 2
source_mean = torch.randn(num_dimensions)  # Random mean vector for each photo
source_cov = torch.eye(num_dimensions)
target_mean = torch.randn(num_dimensions) + 5
target_cov = torch.eye(num_dimensions)

def generate_distribution(mu, cov, num_samples):

    # instantiate Multivariate Gaussian distribution class
    distribution = torch.distributions.MultivariateNormal(mu, cov)
    # generate samples
    samples = distribution.sample((num_samples,))

    return samples

def generate_distributions(mu, cov, num_distributions, num_samples):
    distributions = []
    for i in range(num_distributions):
        distribution = generate_distribution(mu, cov, num_samples)
        distributions.append(distribution)
    distributions = torch.stack(distributions)
    return distributions

source_measures = generate_distributions(source_mean, source_cov, num_distributions, num_samples)
target_measures = generate_distributions(target_mean, target_cov, num_distributions, num_samples)

