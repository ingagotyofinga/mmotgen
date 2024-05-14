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
    # Below works the same as simulate_data, but you can input source and target
    def generate_input_data(self, source, target):
        return torch.cat([source.unsqueeze(1), target.unsqueeze(1)], dim=1)

    def generate_mu0_distributions(self, source_dists):
        source_min = torch.min(source_dists)
        source_max = torch.max(source_dists)
        source_range = np.abs(source_max - source_min)

        num_mu0s = int(np.ceil(0.01 * source_range))
        step = source_range.item() / num_mu0s
        mu0_means_per_dim = [source_min + k * step for k in range(int(num_mu0s))]
        mu0_means_per_dim = torch.stack(mu0_means_per_dim)
        mu0_means_tuple = (mu0_means_per_dim,)*self.num_dimensions
        mu0_means = torch.cartesian_prod(*mu0_means_tuple)
        mu0_std = abs(step/2)

        mu0_distributions = []
        for mean in mu0_means:
            points = []
            for i in mean:
                dim_points = torch.normal(mean = i, std = mu0_std, size=(1,self.num_bins))
                points.append(dim_points)
            points = torch.cat(points,0)
            points = points.transpose(0,1)
            mu0_distributions.append(points)

        mu0_distributions=torch.stack(mu0_distributions)
        # for i in range(num_mu0s):
        #     mu0_samples = torch.normal(mean=mu0_means[i], std=mu0_std, size=(self.num_bins,))
        #     mu0_distributions.append(mu0_samples)
            # plt.plot(mu0_samples, np.zeros_like(mu0_samples), 'x')
        # mu0_distributions = torch.stack(mu0_distributions)
        # mu0 = []
        # for dist in mu0_distributions:
        #     mu0_tuple = (dist,) * num_dimensions
        #     mu0_list = torch.cartesian_prod(*mu0_tuple)
        #     mu0.append(mu0_list)

        return mu0_distributions, step

    def plot_distributions(self, distributions):
        num_plots = distributions.shape[0]
        fig, axs = plt.subplots(num_plots, figsize=(8, 4 * num_plots))
        for i in range(num_plots):
            axs[i].plot(distributions[i], np.zeros_like(distributions[i]), 'x')
            axs[i].set_title(f'Distribution {i + 1}')
        plt.show()


# Example usage
# num_distributions = 10
# num_samples = 100
# num_dimensions = 3

# Means and standard deviations for each cluster in n-dimensional space
# TODO: build these into the function for simulating data
# means = np.random.randint(low=1,high=500, size=(num_distributions, num_dimensions))
# std_devs = np.random.randint(low=1,high=50, size=(num_distributions, num_dimensions))
#
# simulator = DataSimulator(num_distributions, num_samples, num_dimensions)
# source_dists = simulator.generate_source_data(means,std_devs)
# target_dists = simulator.generate_target_data(source_dists)
# input_data = simulator.generate_input_data(source_dists,target_dists)
# mu_data,step = simulator.generate_mu0_distributions(source_dists)

# Visualize the generated data
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for dist in range(len(mu_data)):
#     # ax.scatter(input_data[dist,:,:,0], input_data[dist,:,:,1], input_data[dist,:,:,2])
#     ax.scatter(mu_data[dist,:,0], mu_data[dist,:,1], mu_data[dist,:,2])
#     # ax.scatter(source_dists[dist, :, 0], source_dists[dist, :, 1], source_dists[dist, :, 2], label='Source')
#     # ax.scatter(target_dists[dist, :, 0], target_dists[dist, :, 1], target_dists[dist, :, 2], label='Target')
# # ax.legend()
# plt.title("Sampled $\mu_0$ Distributions")
# plt.grid(True)
# plt.show()
