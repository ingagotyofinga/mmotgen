import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def singlevariable_gaussian_transport_map(source_samples, mu_source=None, sigma_source=None, mu_target=None, sigma_target=None):
    """
    Apply transport map from source to target Gaussian distribution using sample estimators.

    Parameters:
    - source_samples: Samples from the source distribution.
    - mu_source: Mean of the source distribution (or sample mean if None).
    - sigma_source: Standard deviation of the source distribution (or sample std if None).
    - mu_target: Mean of the target distribution (or sample mean if None).
    - sigma_target: Standard deviation of the target distribution (or sample std if None).

    Returns:
    - mapped_samples: Mapped samples in the target distribution.
    """
    if mu_source is None:
        mu_source = source_samples.mean()
    if sigma_source is None:
        sigma_source = source_samples.std()

    # You can similarly handle mu_target and sigma_target using sample estimators

    mapped_samples = mu_target + (sigma_target / sigma_source) * (source_samples - mu_source)
    return mapped_samples

def multivariate_gaussian_transport_map_from_samples(source_samples, mu_target, sigma_target):
    # Calculate the mean and covariance of the source samples
    mu_source = torch.mean(source_samples, dim=0)
    cov_source = torch.matmul((source_samples - mu_source).T, source_samples - mu_source) / (source_samples.size(0) - 1)

    # Calculate the Cholesky decomposition of the covariance matrices
    chol_source = torch.linalg.cholesky(cov_source)
    chol_target = torch.diag(sigma_target).cholesky()

    # Transform the source samples using the optimal transport map
    transport_map = mu_target + torch.matmul(source_samples - mu_source, chol_target.T) @ torch.inverse(chol_source.T)

    return transport_map

# Example usage
mu_source = torch.tensor([2.0, 3.0])
sigma_source = torch.tensor([1.0, 0.5])

mu_target = torch.tensor([5.0, 7.0])
sigma_target = torch.tensor([1.5, 1.0])

# Use float() to convert tensor values to Python floats
source_samples = torch.normal(mean=mu_source.repeat(100, 1), std=sigma_source.repeat(100, 1))

transport_map = multivariate_gaussian_transport_map_from_samples(source_samples, mu_target, sigma_target)

# Plotting
plt.scatter(source_samples[:, 0].numpy(), source_samples[:, 1].numpy(), label='Source Distribution')
plt.scatter(transport_map[:, 0].numpy(), transport_map[:, 1].numpy(), label='Mapped Points')
plt.scatter(mu_target[0].item(), mu_target[1].item(), marker='x', color='red', label='Target Mean')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()
