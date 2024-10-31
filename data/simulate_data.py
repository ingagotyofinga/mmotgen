import torch
import numpy as np
import matplotlib.pyplot as plt


class DataSimulator:
    def __init__(self, num_distributions, num_samples, seed=None):
        self.num_distributions = num_distributions
        self.num_samples = num_samples
        self.seed = seed
        self.data = []

    def generate_data(self, mean, cov):
        if self.seed is not None:
            torch.manual_seed(self.seed)
        self.data = []
        for i in range(self.num_distributions):
            mean_with_randomness = mean + self.add_randomness(mean)
            samples = torch.distributions.MultivariateNormal(mean_with_randomness, cov).sample((self.num_samples,))
            self.data.append(samples)
        self.data = torch.stack(self.data)

    def generate_mu0_distributions(self, cov):
        source_min = torch.min(self.data)
        source_max = torch.max(self.data)
        source_range = torch.abs(source_max - source_min)

        num_mu0s = int(torch.ceil(source_range))
        step = source_range.item() / num_mu0s
        mu0_means_per_dim = [source_min + k * step for k in range(int(num_mu0s))]
        mu0_means_per_dim = torch.stack(mu0_means_per_dim)
        mu0_means_tuple = (mu0_means_per_dim,) * self.source_means.shape[1]
        mu0_means = torch.cartesian_prod(*mu0_means_tuple)
        mu0_std = 1
        # mu0_std = abs(step / 2)

        mu0_distributions = []
        # Assuming the same covariance structure for all

        for mean in mu0_means:
            if mean.dim() == 0:  # Ensure mean is at least 1-dimensional
                mean = mean.unsqueeze(0)
            distribution = torch.distributions.MultivariateNormal(mean, cov)
            samples = distribution.sample((self.num_samples,))
            mu0_distributions.append(samples)

        mu0_distributions = torch.stack(mu0_distributions)
        return mu0_distributions, step

    def add_randomness(self,x):
        '''
        adds uniform (a,b] randomness to x

        Parameters:
        x: torch.Tensor
        a: int
        b: int
        '''

        randomness = torch.randn_like(x)
        return randomness

    def plot_data(self, data, title):
        plt.figure(figsize=(12, 6))
        for i in range(self.num_distributions):
            plt.hist(data[i].numpy(), bins=10, alpha=0.5, edgecolor='black', label=f'Distribution {i + 1}')
        plt.title(title)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_mu0_distributions(self, mu0_distributions, title):
        plt.figure(figsize=(12, 6))
        for i in range(len(mu0_distributions)):
            plt.hist(mu0_distributions[i].numpy(), bins=10, alpha=0.5, edgecolor='black',
                     label=f'Mu0 Distribution {i + 1}')
        plt.title(title)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    # Parameters
    num_distributions = 4
    num_samples = 1000
    num_dimensions = 1
    source_mean = torch.zeros(num_dimensions)
    target_mean = 2*torch.ones(num_dimensions)+5
    covariance = torch.eye(num_dimensions)
    # seed = 42  # Set a seed for reproducibility

    # Create and use the DataSimulator
    generator = DataSimulator(num_distributions, num_samples)
    generator.generate_data(source_mean, covariance)
    source_dists = generator.data
    # # Generate target data
    generator.generate_data(target_mean, covariance)
    target_dists = generator.data

    # Plot the data
    generator.plot_data(source_dists, 'Histogram of Generated 1D Gaussian Source Data')
    generator.plot_data(target_dists, 'Histogram of Generated 1D Gaussian Target Data')

    # Generate mu0 distributions based on source data
    mu0_distributions, step = generator.generate_mu0_distributions(covariance)

    # Plot the mu0 distributions
    generator.plot_mu0_distributions(mu0_distributions, 'Histogram of Mu0 Distributions')

    # Print the shape of mu0_distributions to verify
    print("Mu0 Distributions Shape:", mu0_distributions.shape)
