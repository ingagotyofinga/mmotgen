import torch

# Define constants and initializations
n = 100  # sample size
dim = 2  # dimension of the distributions
m = torch.randn(n, dim, 1)  # source means (n x dim x 1)
q = torch.randn(n, dim, 1) + 9 # target means (n x dim x 1)
Sigma = [torch.eye(dim) for _ in range(n)]  # list of source covariance matrices (dim x dim)
Gamma = [torch.eye(dim) for _ in range(n)]  # list of target covariance matrices (dim x dim)
mu0 = torch.zeros(dim, 1)  # reference mean
Sigma0 = torch.eye(dim)  # reference covariance matrix

print(f"Source Mean:", m.mean(dim=0))
print(f"Target Means:", q.mean(dim=0))


# Wasserstein distance between Gaussians
def wasserstein_distance_gaussian(mu1, Sigma1, mu2, Sigma2):
    mean_dist = torch.norm(mu1 - mu2) ** 2
    Sigma1_sqrt = torch.linalg.cholesky(Sigma1)  # Square root of Sigma1
    inner_term = Sigma1_sqrt @ Sigma2 @ Sigma1_sqrt.T
    trace_term = torch.trace(Sigma1 + Sigma2 - 2 * torch.linalg.cholesky(inner_term))
    return mean_dist + trace_term


# Gaussian Wasserstein-based kernel
def wasserstein_kernel(mu0, Sigma0, mui, Sigmai, bandwidth=1.0):
    w2_squared = wasserstein_distance_gaussian(mu0, Sigma0, mui, Sigmai)
    return torch.exp(-w2_squared / (2 * bandwidth ** 2))


# Compute Y_i = Σ_i^(1/2) * Γ_i * Σ_i^(1/2)
def compute_Y(Sigma_i, Gamma_i):
    Sigma_i_sqrt = torch.linalg.cholesky(Sigma_i)
    return Sigma_i_sqrt @ Gamma_i @ Sigma_i_sqrt.T


# Compute alpha based on current estimate of B
def compute_alpha(B, m, q, mu0, Sigma0, Sigma, bandwidth=1.0, epsilon=0):
    numerator = sum(
        (-B @ m[i] + q[i]) * wasserstein_kernel(mu0, Sigma0, m[i], Sigma[i], bandwidth)
        for i in range(n)
    )
    denominator = sum(wasserstein_kernel(mu0, Sigma0, m[i], Sigma[i], bandwidth) for i in range(n))

    # print("Numerator:", numerator)
    # print("Denominator:", denominator)

    return numerator / (denominator + epsilon)


# Compute B based on current estimate of alpha
def compute_B(alpha, m, q, Sigma, Gamma, mu0, Sigma0, bandwidth=1.0, epsilon=0):
    gradient = 0
    for i in range(n):
        term1 = 2 * (alpha + B @ m[i] - q[i]) @ m[i].T
        term2 = 2 * B @ Sigma[i]

        term3 = B @ Sigma[i] @ B.T + epsilon * torch.eye(B.shape[0]).inverse().sqrt()
        term4 = term3 * (compute_Y(Sigma[i], Gamma[i]) @ B.T + B @ compute_Y(Sigma[i], Gamma[i]).T)

        kernel_value = wasserstein_kernel(mu0, Sigma0, m[i], Sigma[i], bandwidth)

        # print(f"Iteration {i}, Term1:", term1)
        # print(f"Iteration {i}, Term2:", term2)
        # print(f"Iteration {i}, Term3:", term3)
        # print(f"Iteration {i}, Term4:", term4)
        # print(f"Iteration {i}, Kernel Value:", kernel_value)

        gradient += (term1 + term2 - term4) * kernel_value

    # print("Gradient:", gradient)
    return B - learning_rate * gradient


# Initialize B and set optimization parameters
B = torch.eye(dim) + 0.01 * torch.randn(dim, dim)  # initialize B as a random dim x dim matrix
tolerance = 1e-5
max_iter = 100
learning_rate = 1e-5

# Alternating optimization loop
for iteration in range(max_iter):
    # Step 1: Compute alpha with current B
    alpha = compute_alpha(B, m, q, mu0, Sigma0, Sigma)
    print(f"Iteration {iteration}, Alpha:", alpha)

    # Step 2: Update B based on the computed alpha
    B_new = compute_B(alpha, m, q, Sigma, Gamma, mu0, Sigma0)
    print(f"Iteration {iteration}, B_new:", B_new)

    # Check for convergence
    if torch.norm(B_new - B) < tolerance:
        # print(f"Converged after {iteration + 1} iterations.")
        break
    B = B_new

# Final values of alpha and B
print("Final alpha:", alpha)
print("Final B:", B)
