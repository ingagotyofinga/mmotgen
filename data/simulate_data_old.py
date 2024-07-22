import torch
import numpy as np
import matplotlib.pyplot as plt
import geomloss
import matplotlib.colors as mcolors


class DataSimulatorOld:
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

    # def generate_source_data(self, means, std_devs):
    #     """
    #      Generate source data given means and standard deviations for each dimension.
    #
    #      Parameters:
    #      means (list of list of float): Means for each dimension.
    #      std_devs (list of list of float): Standard deviations for each dimension.
    #
    #      Returns:
    #      torch.Tensor: Standardized source distributions.
    #      """
    #     dists = []
    #     for mean, std_dev in zip(means, std_devs):
    #         points = []
    #         for i, j in zip(mean, std_dev):
    #             dim_points = torch.normal(i, j, size=(1, self.num_bins))
    #             points.append(dim_points)
    #         points = torch.cat(points, 0)
    #         points = points.transpose(0, 1)
    #         dists.append(points)
    #
    #     source_dists = torch.stack(dists)
    #     # Standardize source_dists
    #     source_dists = self._standardize(source_dists)
    #     return source_dists

    def generate_source_data (self, mean=0.0, std_dev=1.0):
        """
        Generate n one-dimensional Gaussian distributions each with k samples using PyTorch.

        Parameters:
        n (int): Number of Gaussian distributions.
        k (int): Number of samples in each distribution.
        mean (float): Mean of the Gaussian distributions. Default is 0.0.
        std_dev (float): Standard deviation of the Gaussian distributions. Default is 1.0.

        Returns:
        torch.Tensor: A tensor of shape (n, k) containing samples from Gaussian distributions.
        """
        epsilon = torch.normal(0.0, 1.0, (self.num_distributions,))
        means = mean + epsilon
        distributions = torch.normal(means.unsqueeze(1).expand(self.num_distributions,self.num_bins), std_dev)
        return distributions

    def generate_source_data(self, means, std_devs):
        dists = []
        for mean, std_dev in zip(means, std_devs):
            points = []
            for i, j in zip(mean, std_dev):
                dim_points = torch.normal(i, j, size=(self.num_bins,))
                points.append(dim_points)
            points = torch.stack(points, dim=1)
            dists.append(points)

        source_dists = torch.stack(dists)
        source_dists = self._standardize(source_dists)
        return source_dists

    def generate_target_data(self, dist1, dist2):
        target_dists = []

        for dist in dist1:
            target_dist = self._apply_piecewise_map(dist, dist2)
            target_dists.append(target_dist)

        target_dists = torch.stack(target_dists)
        # target_dists = self._standardize(target_dists)
        return target_dists

    # def generate_target_data(self, source_dists):
    #     """
    #     Generate target data by applying an affine and random map to the source data.
    #
    #     Parameters:
    #     source_dists (torch.Tensor): Source distributions.
    #
    #     Returns:
    #     torch.Tensor: Standardized target distributions.
    #     """
    #
    #     # Generate an affine map
    #     affine_map = torch.randn(self.num_bins, self.num_bins)
    #     affine_translation = torch.randn(self.num_bins, self.num_dimensions)
    #
    #     # Generate a random map
    #     random_map = torch.randn(self.num_bins, self.num_bins)
    #     random_translation = torch.randn(self.num_bins, self.num_dimensions)
    #
    #     # Compose the affine map and the random map
    #     composed_map = torch.matmul(random_map, affine_map)
    #     composed_translation = torch.matmul(random_map, affine_translation) + random_translation
    #
    #     # Standardize composed_map
    #     mean_map = composed_map.mean(dim=0)
    #     std_map = composed_map.std(dim=0)
    #     standardized_composed_map = (composed_map - mean_map) / std_map
    #
    #     # Standardize composed_translation
    #     mean_translation = composed_translation.mean(dim=0)
    #     std_translation = composed_translation.std(dim=0)
    #     standardized_composed_translation = (composed_translation - mean_translation) / std_translation
    #
    #     # Apply the composed map to the source to generate the target
    #     target_dists = torch.matmul(standardized_composed_map, source_dists) + standardized_composed_translation
    #
    #     # Standardize target_dists
    #     target_dists = self._standardize(target_dists)
    #
    #     return target_dists
    #
    # # def generate_target_data(self, source_dists):
    # #     # Define the affine transformation (e.g., rotation matrix)
    # #     theta = np.pi / 6  # 30 degrees rotation for example
    # #     rotation_matrix = self._generate_rotation_matrix(self.num_dimensions, theta)
    # #
    # #     # Define a translation vector to shift the point cloud
    # #     translation_vector = torch.tensor([1.0] * self.num_dimensions)  # Example translation
    # #
    # #     # Apply the affine transformation
    # #     transformed_dists = torch.matmul(source_dists, rotation_matrix) + translation_vector
    # #
    # #     # Add random noise to the transformed distributions
    # #     noise = torch.randn_like(transformed_dists) * 0.1  # Adjust noise level as needed
    # #     target_dists = transformed_dists + noise
    # #
    # #     # Standardize the target distributions
    # #     target_dists = self._standardize(target_dists)
    #
    #     return target_dists

    def generate_input_data(self, source, target):
        return torch.cat([source.unsqueeze(1), target.unsqueeze(1)], dim=1)

    def _standardize(self, data):
        mean = data.mean(dim=0, keepdim=True)
        std = data.std(dim=0, keepdim=True)
        standardized_data = (data - mean) / std
        return standardized_data

    def w2_dist(self, data1, data2):
        sink_loss = geomloss.SamplesLoss(loss='sinkhorn', p=1, blur=0.05)
        wasserstein_distance = sink_loss(data1, data2)
        return wasserstein_distance

    def sigmoid_wasserstein(self, W, theta, epsilon):
        """
        Sigmoid function based on Wasserstein distance.

        Parameters:
        W (float): The Wasserstein distance.
        theta (float): Parameter controlling the steepness of the sigmoid function.
        epsilon (float): Threshold for the Wasserstein distance.

        Returns:
        float: The value of the sigmoid function.
        """
        return 1 / (1 + np.exp(-theta * (W - epsilon)))

    def _apply_piecewise_map(self, dist1, dist2):
        # define T1 for 2d dists
        scaling_factors_1 = torch.tensor([2.0, 2.0], dtype=torch.float32)
        translation_vector_1 = torch.tensor([0.5, -0.5], dtype=torch.float32)

        # define T2 for 2d dists
        scaling_factors_2 = torch.tensor([3.0, 3.0], dtype=torch.float32)
        translation_vector_2 = torch.tensor([-0.5, 0.5], dtype=torch.float32)

        # define epsilon (redundant; it's the same as step)
        eps = 1.5
        a = 0.5
        w2 = self.w2_dist(dist1, dist2).item()  # compute w2 bw mu0 and mui
        s = self.sigmoid_wasserstein(w2, a, eps)  # compute sigmoid function

        print(f"w2: {w2}, s: {s}")  # Debug print

        if s > 0.5:
            transformed_points = dist1 * scaling_factors_1 + translation_vector_1
        else:
            transformed_points = dist1 * scaling_factors_2 + translation_vector_2
        return transformed_points

    def generate_mu0_distributions(self, source_dists):
        source_min = torch.min(source_dists)
        source_max = torch.max(source_dists)
        source_range = np.abs(source_max - source_min)

        num_mu0s = int(np.ceil(source_range))
        step = source_range.item() / num_mu0s
        mu0_means_per_dim = [source_min + 0.8*k * step for k in range(int(0.8*num_mu0s))]
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

    def generate_muR(self):
        samples = np.random.normal(0, 1, (self.num_bins, self.num_dimensions))
        mu0 = torch.tensor(samples, dtype=torch.float32).unsqueeze(0)
        return mu0

    def plot_distributions(self, source_dists, target_dists, mu, muR, eps):
        num_plots = source_dists.shape[0]
        num_mu = mu.shape[0]
        fig, ax = plt.subplots(figsize=(12, 8))

        # Colors for different regions
        color_in_r1 = 'green'
        color_out_r1 = 'red'

        for i in range(num_plots):
            w2 = self.w2_dist(source_dists[i], muR).item()
            if w2 <= eps:
                color = color_in_r1
                label = f'Source {i + 1} in $R_1$'
            else:
                color = color_out_r1
                label = f'Source {i + 1} in complement'

            ax.scatter(source_dists[i, :, 0], source_dists[i, :, 1],
                       color=color, alpha=0.7, label=label if i == 0 else "")
            ax.scatter(target_dists[i, :, 0], target_dists[i, :, 1],
                       color=color, alpha=0.3)

            # Annotate source and target distributions
            x_source, y_source = source_dists[i, :, 0].mean(), source_dists[i, :, 1].mean()
            x_target, y_target = target_dists[i, :, 0].mean(), target_dists[i, :, 1].mean()
            ax.annotate(f'Source {i + 1}', (x_source, y_source), textcoords="offset points", xytext=(0, 10),
                        ha='center',
                        fontsize=11, color='black', fontweight='bold')
            ax.annotate(f'Target {i + 1}', (x_target, y_target), textcoords="offset points", xytext=(0, 10),
                        ha='center',
                        fontsize=11, color='black', fontweight='bold')
        for i in range (num_mu):
            ax.scatter(mu[i, :, 0], mu[i, :, 1],
                       color='blue', alpha=0.1)
            x_mu, y_mu = mu[i, :, 0].mean(), mu[i, :, 1].mean()
            ax.annotate(f'Mu {i + 1}', (x_mu, y_mu), textcoords="offset points", xytext=(0, 10),
                        ha='center',
                        fontsize=11, color='black', fontweight='bold')

        # Plot the boundary
        circle = plt.Circle((0,0), np.sqrt(eps), color='blue', fill=False,
                            linestyle='--')
        ax.add_patch(circle)

        ax.set_title('Source and Mu Distributions with $R_1$ Boundary')
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.show()

# Example usage
if __name__ == "__main__":
    torch.manual_seed(42)
    num_distributions = 100
    num_samples = 100
    num_dimensions = 2

    # Means and standard deviations for each cluster in n-dimensional space
    means = np.random.randint(low=1, high=500, size=(num_distributions, num_dimensions))
    std_devs = np.random.randint(low=1, high=50, size=(num_distributions, num_dimensions))

    simulator = DataSimulatorOld(num_distributions, num_samples, num_dimensions)
    muR = simulator.generate_muR()
    source_dists = simulator.generate_source_data()

    # Plot the distributions
    plt.figure(figsize=(12, 6))

    for i in range(num_distributions):
        plt.hist(source_dists[i].numpy(), bins=30, alpha=0.5, label=f'Distribution {i + 1}')

    plt.title('Generated Gaussian Distributions with Random Error in Means')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    target_dists = simulator.generate_target_data(source_dists)
    # target_dists = simulator.generate_target_data(source_dists,muR[0])
    mu_data, step = simulator.generate_mu0_distributions(source_dists)
    input_data = simulator.generate_input_data(source_dists, target_dists)

    eps = 1.5  # Set epsilon value for the boundary
    # simulator.plot_distributions(source_dists, target_dists, mu_data, muR[0], eps)

    for i in range(len(mu_data)):
    # Plot source distribution with lighter shade
        plt.scatter(mu_data[i, :, 0], mu_data[i, :, 1])

    plt.show()


    # # Prepare data for visualization
    # data = [source_dists, target_dists]
    # labels = ["Source", "Target"]
    #
    # # # Define base colors for each distribution with lighter shades
    # # source_colors = ['#66c2a5', '#8da0cb', '#fc8d62', '#e78ac3', '#a6d854']
    # # target_colors = ['#1b9e77', '#7570b3', '#d95f02', '#e7298a', '#66a61e']
    #
    # # Define base colors for each distribution
    # base_colors = plt.cm.summer(np.linspace(0, 1, num_distributions))
    #
    # plt.figure(figsize=(12, 8))
    # for i in range(num_distributions):
    #     # Plot source distribution with lighter shade
    #     plt.scatter(source_dists[i, :, 0], source_dists[i, :, 1],
    #                 color=base_colors[i], alpha=0.3,
    #                 label=f'Source {i + 1}' if i == 0 else "")
    #     # Plot target distribution with darker shade of the same color
    #     plt.scatter(target_dists[i, :, 0], target_dists[i, :, 1],
    #                 color=base_colors[i], alpha=0.7,
    #                 label=f'Target {i + 1}' if i == 0 else "")
    #     # Annotate source distributions at the center
    #     x_source, y_source = source_dists[i, :, 0].mean(), source_dists[i, :, 1].mean()
    #     plt.annotate(f'Source {i + 1}', (x_source, y_source), textcoords="offset points", xytext=(0, 10), ha='center',
    #                  fontsize=11, color='black', fontweight='bold')
    #     # Annotate target distributions at the center
    #     x_target, y_target = target_dists[i, :, 0].mean(), target_dists[i, :, 1].mean()
    #     plt.annotate(f'Target {i + 1}', (x_target, y_target), textcoords="offset points", xytext=(0, 10), ha='center',
    #                  fontsize=11, color='black', fontweight='bold')
    #
    # plt.title('Source and Target Distributions (2D Point Clouds)')
    # plt.xlabel('$x_1$')
    # plt.ylabel('$x_2$')
    # plt.grid(True, linestyle='--', alpha=0.7)
    # # plt.legend([plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=base_colors[0], markersize=10, alpha=0.3),
    # #             plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=base_colors[0], markersize=10, alpha=0.7)],
    # #            ['Source', 'Target'], loc='upper right')
    # plt.show()
    # # ax.set_zlabel('Dimension 3')
    # # ax.legend()
