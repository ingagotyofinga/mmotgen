import torch
import numpy as np
import matplotlib.pyplot as plt


class DataSimulator:
    def __init__(self, num_distributions, num_samples, source_means, target_means, covariances, seed=None):
        self.num_distributions = num_distributions
        self.num_samples = num_samples
        self.source_means = torch.tensor(source_means, dtype=torch.float32)
        self.target_means = torch.tensor(target_means, dtype=torch.float32)
        self.covariances = torch.tensor(covariances, dtype=torch.float32)
        self.seed = seed
        self.data = []
        self.target_data = []

    def generate_data(self, source_means_std):
        if self.seed is not None:
            torch.manual_seed(self.seed)
        self.data = []

        # Add randomness to source means
        source_means_with_noise = self.add_randomness_to_means(self.source_means, source_means_std)

        for i in range(self.num_distributions):
            mean = source_means_with_noise[i]
            cov = self.covariances[i]
            samples = torch.distributions.MultivariateNormal(mean, cov).sample((self.num_samples,))
            self.data.append(samples)
        self.data = torch.stack(self.data)

    def add_randomness_to_means(self, means, std_dev):
        if self.seed is not None:
            torch.manual_seed(self.seed)
        random_perturbation = torch.normal(0, std_dev, means.shape)
        return means + random_perturbation

    def generate_target_data(self, source_means_std, target_means_std):
        source_means_with_noise = self.add_randomness_to_means(self.source_means, source_means_std)
        target_means_with_noise = self.add_randomness_to_means(self.target_means, target_means_std)
        self.target_data = []

        for i in range(self.num_distributions):
            source_data = self.data[i]
            target_data = (source_data - source_means_with_noise[i]) + target_means_with_noise[i]
            self.target_data.append(target_data)

        self.target_data = torch.stack(self.target_data)
        return self.target_data

    def generate_input_data(self):
        return torch.cat([self.data.unsqueeze(1), self.target_data.unsqueeze(1)], dim=1)

    def generate_mu0_distributions(self):
        source_min = torch.min(self.data)
        source_max = torch.max(self.data)
        source_range = torch.abs(source_max - source_min)

        num_mu0s = int(torch.ceil(source_range / 2))
        step = source_range.item() / num_mu0s
        mu0_means_per_dim = [source_min + k * step for k in range(int(num_mu0s))]
        mu0_means_per_dim = torch.stack(mu0_means_per_dim)
        mu0_means_tuple = (mu0_means_per_dim,) * self.source_means.shape[1]
        mu0_means = torch.cartesian_prod(*mu0_means_tuple)
        mu0_std = abs(step)

        mu0_distributions = []
        cov = torch.eye(self.source_means.shape[1]) * mu0_std  # Assuming the same covariance structure for all

        for mean in mu0_means:
            if mean.dim() == 0:  # Ensure mean is at least 1-dimensional
                mean = mean.unsqueeze(0)
            distribution = torch.distributions.MultivariateNormal(mean, cov)
            samples = distribution.sample((self.num_samples,))
            mu0_distributions.append(samples)

        mu0_distributions = torch.stack(mu0_distributions)
        return mu0_distributions, step

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
    num_distributions = 10
    num_samples = 100
    source_means = [[0] for i in range(num_distributions)]
    target_means = [[1] for i in range(num_distributions)]
    covariances = [[[1]] for _ in range(num_distributions)]
    seed = 42  # Set a seed for reproducibility

    # Create and use the DataSimulator
    generator = DataSimulator(num_distributions, num_samples, source_means, target_means, covariances, seed)

    # Standard deviation for the noise added to source means
    source_means_std = 2.0

    generator.generate_data(source_means_std)
    source_dists = generator.data  # Assign the generated data to source_dists

    # Generate target data
    target_means_std = 2.0  # Standard deviation for the noise added to target means
    target_data = generator.generate_target_data(source_means_std, target_means_std)

    # Plot the data
    generator.plot_data(generator.data, 'Histogram of Generated 1D Gaussian Source Data')
    generator.plot_data(target_data, 'Histogram of Generated 1D Gaussian Target Data')

    # Generate mu0 distributions based on source data
    mu0_distributions, step = generator.generate_mu0_distributions()

    # Plot the mu0 distributions
    generator.plot_mu0_distributions(mu0_distributions, 'Histogram of Mu0 Distributions')

    # Print the shape of mu0_distributions to verify
    print("Mu0 Distributions Shape:", mu0_distributions.shape)
