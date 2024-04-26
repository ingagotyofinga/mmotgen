import torch
import numpy as np
import matplotlib.pyplot as plt


class DataSimulator:
    def __init__(self, num_distributions, num_bins, num_dimensions, seed=None):
        self.num_distributions = num_distributions
        self.num_bins = num_bins
        self.num_dimensions = num_dimensions
        self.seed = seed
    def simulate_mu(self):
        mu_distributions = []
        for _ in range(self.num_distributions):
            mu_mean = torch.randn(1).item()
            mu_std = torch.abs(torch.randn(1)).item()
            mu_samples = torch.normal(mean=mu_mean, std=mu_std, size=(self.num_bins,))
            mu_distributions.append(mu_samples)
        return torch.stack(mu_distributions)

    def simulate_nu(self):
        nu_distributions = []
        for _ in range(self.num_distributions):
            nu_mean = torch.randn(1).item()
            nu_std = torch.abs(torch.randn(1)).item()
            nu_samples = torch.normal(mean=nu_mean, std=nu_std, size=(self.num_bins,))
            nu_distributions.append(nu_samples)
        return torch.stack(nu_distributions)

    def simulate_data(self):
        source_dists = self.simulate_mu()
        target_dists = self.simulate_nu()
        input_data = torch.cat([source_dists.unsqueeze(1), target_dists.unsqueeze(1)], dim=1)
        return input_data

    def generate_source_data(self, means, std_devs):
        dists = []
        for mean, std_dev in zip(means, std_devs):
            points = []
            for i,j in zip(mean,std_dev):
                dim_points = torch.normal(i,j, size=(1,self.num_bins))
                points.append(dim_points)
            points = torch.cat(points,0)
            points = points.transpose(0,1)
            dists.append(points)

        return torch.stack(dists)

    def generate_target_data(self, source_dists):
        # Generate an affine map
        affine_map = torch.randn(self.num_bins, self.num_bins)
        affine_translation = torch.randn(self.num_bins,self.num_dimensions)

        # Generate a random map
        random_map = torch.randn(self.num_bins, self.num_bins)
        random_translation = torch.randn(self.num_bins, self.num_dimensions)

        # Compose the affine map and the random map
        composed_map = torch.matmul(random_map, affine_map)
        composed_translation = torch.matmul(random_map, affine_translation) + random_translation

        # Standardize composed_map
        mean_map = composed_map.mean(dim=0)
        std_map = composed_map.std(dim=0)
        standardized_composed_map = (composed_map - mean_map) / std_map

        # Standardize composed_translation
        mean_translation = composed_translation.mean(dim=0)
        std_translation = composed_translation.std(dim=0)
        standardized_composed_translation = (composed_translation - mean_translation) / std_translation

        # Apply the composed map to the source to generate the target
        target_dists=torch.matmul(composed_map, source_dists) + composed_translation


        return target_dists

    def generate_mu0_distributions(self, source_dists):
        source_min = torch.min(source_dists)
        source_range = np.abs(torch.max(source_dists) - source_min)
        num_mu0s = int(np.ceil(2 * source_range))
        step = source_range / num_mu0s
        mu0_means = [source_min + k * step for k in range(int(num_mu0s))]
        mu0_std = torch.abs(step)

        mu0_distributions = []
        for i in range(num_mu0s):
            mu0_samples = torch.normal(mean=mu0_means[i], std=mu0_std, size=(self.num_bins,))
            mu0_distributions.append(mu0_samples)
            plt.plot(mu0_samples, np.zeros_like(mu0_samples), 'x')

        return mu0_distributions, step

    def plot_distributions(self, distributions):
        num_plots = distributions.shape[0]
        fig, axs = plt.subplots(num_plots, figsize=(8, 4 * num_plots))
        for i in range(num_plots):
            axs[i].plot(distributions[i], np.zeros_like(distributions[i]), 'x')
            axs[i].set_title(f'Distribution {i + 1}')
        plt.show()


# Example usage
num_distributions = 10
num_samples = 100
num_dimensions = 3

# Means and standard deviations for each cluster in n-dimensional space
# TODO: build these into the function for simulating data
means = np.random.randint(low=1,high=500, size=(num_distributions, num_dimensions))
std_devs = np.random.randint(low=1,high=50, size=(num_distributions, num_dimensions))

simulator = DataSimulator(num_distributions, num_samples, num_dimensions)
source_dists = simulator.generate_source_data(means,std_devs)
# TODO: scaling of the target data is off now
target_dists = simulator.generate_target_data(source_dists)

# Visualize the generated data
# TODO: visualization script for the data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for dist in range(len(source_dists)):
    # ax.scatter(source_dists[dist, :, 0], source_dists[dist, :, 1], source_dists[dist, :, 2], label='Source')
    ax.scatter(target_dists[dist, :, 0], target_dists[dist, :, 1], target_dists[dist, :, 2], label='Target')
# ax.legend()
plt.title("Sampled Target Distributions")
plt.grid(True)
plt.show()
