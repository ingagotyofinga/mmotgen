import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(0)
# Define constants
dim = 2  # Dimension of the distributions
theta = torch.tensor(torch.pi / 4)  # 45 degrees rotation
s = 2.0  # Scaling factor

# Define a rotation + scaling matrix A and the translation vector b
A = s * torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                      [torch.sin(theta), torch.cos(theta)]])
b = torch.tensor([[5.0], [3.0]])  # Translation vector

# Function to compute expected B using the formula
def compute_expected_B(Sigma, Gamma):
    Sigma_reg = Sigma * torch.eye(dim)  # Regularization term
    Sigma_sqrt = torch.linalg.cholesky(Sigma_reg)
    Sigma_inv_sqrt = torch.linalg.inv(Sigma_sqrt)
    middle_term = torch.linalg.cholesky(Sigma_sqrt @ Gamma @ Sigma_sqrt.T)
    return Sigma_inv_sqrt @ middle_term @ Sigma_inv_sqrt.T

# Compute expected alpha using the formula: alpha = q.mean - B * m.mean
def compute_expected_alpha(B, m, q):
    m_mean = m.mean(dim=0)
    q_mean = q.mean(dim=0)
    return q_mean - B @ m_mean

# Wasserstein distance between Gaussians
def wasserstein_distance_gaussian(mu1, Sigma1, mu2, Sigma2):
    mean_dist = torch.norm(mu1 - mu2) ** 2
    Sigma1_sqrt = torch.linalg.cholesky(Sigma1 * torch.eye(dim))
    inner_term = Sigma1_sqrt @ Sigma2 @ Sigma1_sqrt.T
    trace_term = torch.trace(Sigma1 + Sigma2 - 2 * torch.linalg.cholesky(inner_term))
    return mean_dist + trace_term

# Gaussian Wasserstein-based kernel
def wasserstein_kernel(mu0, Sigma0, mui, Sigmai, bandwidth=1.0):
    w2_squared = wasserstein_distance_gaussian(mu0, Sigma0, mui, Sigmai)
    return torch.exp(-w2_squared / (2 * bandwidth ** 2))

# Compute alpha based on current estimate of B
def compute_alpha(B, m, q, mu0, Sigma0, Sigma, bandwidth=1.0):
    weights = torch.tensor([wasserstein_kernel(mu0, Sigma0, m[i], Sigma[i], bandwidth) for i in range(len(m))])
    weighted_sums = torch.sum((-B @ m + q) * weights[:, None, None], dim=0)
    weights_sum = torch.sum(weights)
    return weighted_sums / weights_sum

# Compute loss
def compute_loss(B, a, m, q, Sigma, Gamma, mu0, Sigma0, bandwidth=1.0):
    loss = 0
    for i in range(len(m)):
        m_i_transformed = B @ m[i] + a
        Sigma_i_transformed = B @ Sigma[i] @ B.T
        W2_distance = wasserstein_distance_gaussian(m_i_transformed, Sigma_i_transformed, q[i], Gamma[i])
        kernel_value = wasserstein_kernel(mu0, Sigma0, m[i], Sigma[i], bandwidth)
        loss += W2_distance * kernel_value
    return loss / len(m)

# Sample sizes to evaluate
n_values = [10, 20, 50, 100, 200]
num_repetitions = 1  # Number of repetitions for each sample size
avg_errors_B = []
avg_errors_alpha = []

for n in n_values:
    errors_B = []
    errors_alpha = []

    for _ in range(num_repetitions):
        # Initialize data based on current sample size n
        E = 1e-2 * torch.randn(dim, dim)
        err = 1e-2 * torch.randn(n, dim, 1)
        m = torch.zeros(n, dim, 1) + torch.randn(n, dim, 1)  # Source means
        q = A @ m + b + err  # Target means
        Sigma = [torch.eye(dim) for _ in range(n)]  # Source covariance matrices
        Gamma = [A @ sigma @ A.T for sigma in Sigma]  # Target covariance matrices
        mu0 = torch.randn(dim, 1) # Reference mean
        Sigma0 = torch.eye(dim)  # Reference covariance matrix

        # Initialize B and set optimization parameters
        B = (torch.eye(dim) + 0.01 * torch.randn(dim, dim)).detach().requires_grad_(True)
        learning_rate = 1e-1
        optimizer = optim.Adam([B], lr=learning_rate)

        # Variables to store B and alpha convergence results
        B_expected = A
        alpha_expected = b

        # Alternating optimization loop
        for iteration in range(200):  # Reduced max_iter for efficiency
            # Compute alpha
            alpha = compute_alpha(B, m, q, mu0, Sigma0, Sigma)

            optimizer.zero_grad()  # Reset gradients

            # Define and backpropagate the loss
            loss = compute_loss(B, alpha, m, q, Sigma, Gamma, mu0, Sigma0)
            loss.backward()

            # Update B without gradient clipping
            optimizer.step()

        # Store the final statistical errors for B and alpha
        errors_B.append(torch.norm(B - B_expected).item())
        errors_alpha.append(torch.norm(alpha - alpha_expected).item())

    # Average the errors over all repetitions for this sample size n
    avg_errors_B.append(np.mean(errors_B))
    avg_errors_alpha.append(np.mean(errors_alpha))

    # Print the averaged results after each sample size
    print(f"\nAveraged Results for n = {n}:")
    print(f"Averaged Statistical Error for B: ||B_expected - B|| = {avg_errors_B[-1]}")
    print(f"Averaged Statistical Error for alpha: ||alpha_expected - alpha|| = {avg_errors_alpha[-1]}")

# Plotting averaged errors as a function of n
plt.figure(figsize=(10, 5))

# Plot for averaged error in B
plt.subplot(1, 2, 1)
plt.plot(n_values, avg_errors_B, marker='o', label="Averaged ||B_expected - B||")
plt.xlabel("Sample Size (n)")
plt.ylabel("Averaged Final Statistical Error for B")
plt.title("Averaged Final Error in B vs. Sample Size")
plt.xscale("log")
plt.yscale("log")
plt.legend()

# Plot for averaged error in alpha
plt.subplot(1, 2, 2)
plt.plot(n_values, avg_errors_alpha, marker='o', label="Averaged ||alpha_expected - alpha||")
plt.xlabel("Sample Size (n)")
plt.ylabel("Averaged Final Statistical Error for alpha")
plt.title("Averaged Final Error in alpha vs. Sample Size")
plt.xscale("log")
plt.yscale("log")
plt.legend()

plt.tight_layout()
plt.show()
