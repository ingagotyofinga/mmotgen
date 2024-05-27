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
        """
         Generate source data given means and standard deviations for each dimension.

         Parameters:
         means (list of list of float): Means for each dimension.
         std_devs (list of list of float): Standard deviations for each dimension.

         Returns:
         torch.Tensor: Standardized source distributions.
         """
        dists = []
        for mean, std_dev in zip(means, std_devs):
            points = []
            for i, j in zip(mean, std_dev):
                dim_points = torch.normal(i, j, size=(1, self.num_bins))
                points.append(dim_points)
            points = torch.cat(points, 0)
            points = points.transpose(0, 1)
            dists.append(points)

        source_dists = torch.stack(dists)
        # Standardize source_dists
        source_dists = self._standardize(source_dists)
        return source_dists

    def generate_target_data(self, source_dists):
        """
        Generate target data by applying an affine and random map to the source data.

        Parameters:
        source_dists (torch.Tensor): Source distributions.

        Returns:
        torch.Tensor: Standardized target distributions.
        """

        # Generate an affine map
        affine_map = torch.randn(self.num_bins, self.num_bins)
        affine_translation = torch.randn(self.num_bins, self.num_dimensions)

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
        target_dists = torch.matmul(standardized_composed_map, source_dists) + standardized_composed_translation

        # Standardize target_dists
        target_dists = self._standardize(target_dists)

        return target_dists

    def _standardize(self, data):
        mean = data.mean(dim=0, keepdim=True)
        std = data.std(dim=0, keepdim=True)
        standardized_data = (data - mean) / std
        return standardized_data

    def generate_input_data(self, source, target):
        return torch.cat([source.unsqueeze(1), target.unsqueeze(1)], dim=1)

    def generate_mu0_distributions(self, source_dists):
        source_min = torch.min(source_dists)
        source_max = torch.max(source_dists)
        source_range = np.abs(source_max - source_min)

        num_mu0s = int(np.ceil(source_range))
        step = source_range.item() / num_mu0s
        mu0_means_per_dim = [source_min + k * step for k in range(int(num_mu0s))]
        mu0_means_per_dim = torch.stack(mu0_means_per_dim)
        mu0_means_tuple = (mu0_means_per_dim,) * self.num_dimensions
        mu0_means = torch.cartesian_prod(*mu0_means_tuple)
        mu0_std = abs(step / 2)

        mu0_distributions = []
        for mean in mu0_means:
            points = []
            for i in mean:
                dim_points = torch.normal(mean=i, std=mu0_std, size=(1, self.num_bins))
                points.append(dim_points)
            points = torch.cat(points, 0)
            points = points.transpose(0, 1)
            mu0_distributions.append(points)

        mu0_distributions = torch.stack(mu0_distributions)
        mu0_distributions = self._standardize(mu0_distributions)
        return mu0_distributions, step

    def plot_distributions(self, distributions):
        num_plots = distributions.shape[0]
        fig, axs = plt.subplots(num_plots, figsize=(8, 4 * num_plots))
        for i in range(num_plots):
            axs[i].plot(distributions[i], np.zeros_like(distributions[i]), 'x')
            axs[i].set_title(f'Distribution {i + 1}')
        plt.show()

# Example usage
if __name__ == "__main__":
    num_distributions = 10
    num_samples = 100
    num_dimensions = 3

    # Means and standard deviations for each cluster in n-dimensional space
    means = np.random.randint(low=1,high=500, size=(num_distributions, num_dimensions))
    std_devs = np.random.randint(low=1,high=50, size=(num_distributions, num_dimensions))

    simulator = DataSimulator(num_distributions, num_samples, num_dimensions)
    source_dists = simulator.generate_source_data(means,std_devs)
    target_dists = simulator.generate_target_data(source_dists)
    input_data = simulator.generate_input_data(source_dists,target_dists)
    mu_data,step = simulator.generate_mu0_distributions(source_dists)

    data = [source_dists, target_dists]
    labels = ["Dataset 1", "Dataset 2"]
    colors = ['blue', 'red']
    # Visualize the generated data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for idx, dataset in enumerate(data):
        color = colors[idx] if colors else None
        for i in range(dataset.shape[0]):
            ax.scatter(dataset[i, :, 0], dataset[i, :, 1], dataset[i, :, 2],
                       label=f'{labels[idx]} Distribution {i + 1}' if labels else f'Distribution {i + 1}', color=color)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    # ax.legend()
    plt.show()
