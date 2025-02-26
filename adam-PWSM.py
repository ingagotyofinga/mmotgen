import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

# Define constants
DIM = 2
THETA = torch.tensor(torch.pi / 4)  # 45 degrees rotation
SCALING_FACTOR = 1.0

# Define transformation matrices and translation vectors
A1 = -1 * torch.eye(DIM)
b1 = torch.tensor([[1.0], [-12.0]])

A2 = SCALING_FACTOR * torch.tensor([[torch.cos(THETA), -torch.sin(THETA)],
                                    [torch.sin(THETA), torch.cos(THETA)]])
b2 = torch.tensor([[5.0], [3.0]])

# Compute Wasserstein distance between Gaussian distributions
def wasserstein_distance_gaussian(mu1, Sigma1, mu2, Sigma2):
    mean_dist = torch.norm(mu1 - mu2) ** 2
    Sigma1_sqrt = torch.linalg.cholesky(Sigma1 * torch.eye(DIM))
    inner_term = Sigma1_sqrt @ Sigma2 @ Sigma1_sqrt.T
    trace_term = torch.trace(Sigma1 + Sigma2 - 2 * torch.linalg.cholesky(inner_term))
    return mean_dist + trace_term

# Gaussian Wasserstein-based kernel
def wasserstein_kernel(mu0, Sigma0, mui, Sigmai, bandwidth=1.0):
    w2_squared = wasserstein_distance_gaussian(mu0, Sigma0, mui, Sigmai)
    return torch.exp(-w2_squared / (2 * bandwidth ** 2))

# Compute alpha based on the current estimate of B
def compute_alpha(B, m, q, mu0, Sigma0, Sigma, bandwidth=1.0):
    weights = torch.stack([wasserstein_kernel(mu0, Sigma0, m[i], Sigma[i], bandwidth) for i in range(len(m))])
    weighted_sums = torch.sum((-B @ m + q) * weights[:, None, None], dim=0)
    weights_sum = torch.sum(weights)
    return weighted_sums / (weights_sum + 1e-6)

# Compute the Wasserstein loss
def compute_loss(B, a, m, q, Sigma, Gamma, mu0, Sigma0, bandwidth=1.0):
    loss = 0
    for i in range(len(m)):
        m_i_transformed = B @ m[i] + a
        Sigma_i_transformed = B @ Sigma[i] @ B.T
        W2_distance = wasserstein_distance_gaussian(m_i_transformed, Sigma_i_transformed, q[i], Gamma[i])
        kernel_value = wasserstein_kernel(mu0, Sigma0, m[i], Sigma[i], bandwidth)
        loss += W2_distance * kernel_value
    return loss / len(m)

# Main experiment
def run_experiment(n_values, num_repetitions, max_iter=200, learning_rate=1e-1):
    avg_errors_B = []
    avg_errors_alpha = []

    for n in n_values:
        errors_B = []
        errors_alpha = []

        for _ in range(num_repetitions):
            # Generate data
            m, Sigma, q, Gamma = generate_data(n)

            # Initialize reference distribution
            mu0 = 6 + torch.randn(DIM, 1)
            Sigma0 = torch.eye(DIM)

            # Initialize B and optimizer
            B = (torch.eye(DIM) + 0.01 * torch.randn(DIM, DIM)).detach().requires_grad_(True)
            optimizer = optim.Adam([B], lr=learning_rate)

            # Expected values
            B_expected = A2
            alpha_expected = b2

            # Optimization loop
            for _ in range(max_iter):
                alpha = compute_alpha(B, m, q, mu0, Sigma0, Sigma)
                optimizer.zero_grad()
                loss = compute_loss(B, alpha, m, q, Sigma, Gamma, mu0, Sigma0)
                loss.backward()
                optimizer.step()

            # Store errors
            errors_B.append(torch.norm(B - B_expected).item())
            errors_alpha.append(torch.norm(alpha - alpha_expected).item())

        # Average errors
        avg_errors_B.append(np.mean(errors_B))
        avg_errors_alpha.append(np.mean(errors_alpha))

        # Print results for the current n
        print(f"\nAveraged Results for n = {n}:")
        print(f"Averaged Statistical Error for B: ||B_expected - B|| = {avg_errors_B[-1]}")
        print(f"Averaged Statistical Error for alpha: ||alpha_expected - alpha|| = {avg_errors_alpha[-1]}")
        print(f"Computed B {B} and alpha {alpha}")

    return avg_errors_B, avg_errors_alpha

# Generate data for the experiment
def generate_data(n):
    c1 = torch.zeros(DIM, 1) # centroid 1
    c2 = torch.zeros(DIM, 1) + 10 # centroid 2
    m1 = torch.zeros(n // 2, DIM, 1) + torch.randn(n // 2, DIM, 1)
    m2 = 10 + torch.zeros(n // 2, DIM, 1) + torch.randn(n // 2, DIM, 1)

    # # Convert to 2D arrays for plotting
    # m1 = m1.squeeze().numpy()
    # m2 = m2.squeeze().numpy()
    #
    # # Plot
    # plt.figure(figsize=(8, 6))
    # plt.scatter(m1[:, 0], m1[:, 1], label='m1 (Cluster near 0)', color='blue')
    # plt.scatter(m2[:, 0], m2[:, 1], label='m2 (Cluster near 20)', color='red')
    # plt.plot([c2[0],c1[0]], [c1[0],c2[0]], label='c1', color='blue')
    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')
    # plt.title('Scatter Plot of Generated Clusters m1 and m2')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    m = torch.cat((m1, m2), dim=0)
    Sigma = [torch.eye(DIM) for _ in range(n)]

    q, Gamma = [], []
    for i in range(n):
        wd = wasserstein_distance_gaussian(m[i], Sigma[i], torch.zeros(DIM), torch.eye(DIM)).item()
        if wd < 100:
            q.append(A1 @ m[i] + b1)
            Gamma.append(A1 @ Sigma[i] @ A1.T)
        else:
            q.append(A2 @ m[i] + b2)
            Gamma.append(A2 @ Sigma[i] @ A2.T)
    return m, Sigma, torch.stack(q), Gamma

# Run and plot results
n_values = [10, 20, 50, 100, 200]
num_repetitions = 10
avg_errors_B, avg_errors_alpha = run_experiment(n_values, num_repetitions)

# Plotting
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(n_values, avg_errors_B, marker='o', label="Averaged ||B_expected - B||")
plt.xlabel("Sample Size (n)")
plt.ylabel("Averaged Error for B")
plt.title("Error in B vs. Sample Size")
# plt.xscale("log")
# plt.yscale("log")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(n_values, avg_errors_alpha, marker='o', label="Averaged ||alpha_expected - alpha||")
plt.xlabel("Sample Size (n)")
plt.ylabel("Averaged Error for alpha")
plt.title("Error in alpha vs. Sample Size")
plt.xscale("log")
plt.yscale("log")
plt.legend()

plt.tight_layout()
plt.show()
