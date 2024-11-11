import torch
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(0)
# Define constants and initializations
n = 500 # sample size
dim = 2  # dimension of the distributions
theta = torch.tensor(torch.pi / 4)  # 45 degrees rotation
s = 2.0  # scaling factor

# Define a rotation + scaling matrix A
A = s * torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                      [torch.sin(theta), torch.cos(theta)]])
b = torch.tensor([[5.0], [3.0]])  # Translation vector

E = 1e-2*torch.randn(dim, dim)
# err = [E @ E.T for _ in range(n)]
err = 1e-2*torch.randn(n, dim, 1)
m = torch.zeros(n, dim, 1) + torch.randn(n,dim,1)  # source means (n x dim x 1)
q = A @ m + b + err # target means (n x dim x 1)
Sigma = [torch.eye(dim) for _ in range(n)]  # source covariance matrices (dim x dim)
# Gamma = [A @ sigma @ A.T + e for sigma, e in zip(Sigma, err)] # target covariance matrices (dim x dim)
Gamma = [A @ sigma @ A.T for sigma in Sigma]
mu0 = torch.randn(dim, 1)  # reference mean
Sigma0 = torch.eye(dim)  # reference covariance matrix


# Function to compute expected B using the formula: Σ^(-1/2) * (Σ^(1/2) Γ Σ^(1/2)) Σ^(-1/2)
def compute_expected_B(Sigma, Gamma):
    Sigma_reg = Sigma + 1e-6 * torch.eye(dim)  # Regularization term added
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
    Sigma1_sqrt = torch.linalg.cholesky(Sigma1 + 1e-6 * torch.eye(dim))  # Regularized Sigma1
    inner_term = Sigma1_sqrt @ Sigma2 @ Sigma1_sqrt.T
    trace_term = torch.trace(Sigma1 + Sigma2 - 2 * torch.linalg.cholesky(inner_term))
    return mean_dist + trace_term


# Gaussian Wasserstein-based kernel
def wasserstein_kernel(mu0, Sigma0, mui, Sigmai, bandwidth=1.0):
    w2_squared = wasserstein_distance_gaussian(mu0, Sigma0, mui, Sigmai)
    return torch.exp(-w2_squared / (2 * bandwidth ** 2))


# Compute alpha based on current estimate of B
def compute_alpha(B, m, q, mu0, Sigma0, Sigma, bandwidth=1.0):
    weights = torch.tensor([wasserstein_kernel(mu0, Sigma0, m[i], Sigma[i], bandwidth) for i in range(n)])
    weighted_sums = torch.sum((-B @ m + q) * weights[:, None, None], dim=0)
    weights_sum = torch.sum(weights)
    return weighted_sums / (weights_sum + 1e-6)  # Small epsilon for stability


# Compute loss
def compute_loss(B, a, m, q, Sigma, Gamma, mu0, Sigma0, bandwidth=1.0):
    loss = 0
    for i in range(len(m)):
        m_i_transformed = B @ m[i] + a
        Sigma_i_transformed = B @ Sigma[i] @ B.T
        W2_distance = wasserstein_distance_gaussian(m_i_transformed, Sigma_i_transformed, q[i], Gamma[i])
        kernel_value = wasserstein_kernel(mu0, Sigma0, m[i], Sigma[i], bandwidth)
        loss += W2_distance * kernel_value
    return loss/n


# Initialize B and set optimization parameters
B = (torch.eye(dim) + 0.01 * torch.randn(dim, dim)).detach().requires_grad_(True)
tolerance = 1e-8
max_iter = 400
learning_rate = 1e-1  # Higher learning rate for faster convergence

# Plateau detection parameters
plateau_threshold = 1e-22  # Minimum change in loss considered as progress
plateau_patience = 10  # Number of iterations to check for plateau

# Define optimizer
optimizer = optim.Adam([B], lr=learning_rate)

# Variables to store alpha and B values
alpha_values, B_norms, difference_norms, loss_values = [], [], [], []

# Alternating optimization loop
for iteration in range(max_iter):
    # Step 1: Compute alpha
    alpha = compute_alpha(B, m, q, mu0, Sigma0, Sigma)
    alpha_values.append(alpha.clone().detach().mean().item())

    optimizer.zero_grad()  # Reset gradients

    # Step 2: Define and backpropagate the loss
    loss = compute_loss(B, alpha, m, q, Sigma, Gamma, mu0, Sigma0, bandwidth=1.0)
    loss_values.append(loss.item())  # Store the loss value for plotting

    # Monitoring the loss value
    if iteration % 10 == 0:
        print(f"Iteration {iteration}, Loss: {loss.item()}")

    # Plateau detection
    if iteration > plateau_patience:
        recent_losses = loss_values[-plateau_patience:]
        avg_loss_change = abs(recent_losses[-1] - recent_losses[0]) / plateau_patience
        if avg_loss_change < plateau_threshold:
            print(f"Stopping early at iteration {iteration} due to plateau (average change < {plateau_threshold}).")
            break

    if torch.isnan(loss) or torch.isinf(loss):
        print("Detected NaN or Inf in loss. Exiting.")
        break

    loss.backward()

    # Step 3: Gradient clipping
    torch.nn.utils.clip_grad_norm_([B], max_norm=1.0)
    optimizer.step()  # Update B

    # Compute expected B and norm differences for convergence
    B_norms.append(torch.norm(B).item())
    B_expected = A
    difference_norms.append(torch.norm(B - B_expected).item())

    # Convergence check
    if difference_norms[-1] < tolerance:
        print(f"Converged after {iteration + 1} iterations.")
        break

# Calculate expected B and alpha based on the final optimized values
B_expected = A  # Using the reference covariance matrix
alpha_expected = b

print("m mean:", m.mean(dim=0))
print("q mean:", q.mean(dim=0))
print("Sigma:", Sigma[0])
print("Gamma:", Gamma[0])


# Evaluate results
print("\nTesting Results:")
print("Expected alpha:", alpha_expected)
print("Expected B:", B_expected)
print("Estimated alpha:", alpha)
print("Estimated B:", B)


# Results evaluation and plotting
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.plot(alpha_values, label="Alpha (Mean)")
plt.xlabel("Iteration")
plt.ylabel("Mean of Alpha")
plt.title("Mean of Alpha over Iterations")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(B_norms, label="Norm of B")
plt.xlabel("Iteration")
plt.ylabel("B Norm")
plt.title("Norm of B over Iterations")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(difference_norms, label="||B_expected - B||")
plt.xlabel("Iteration")
plt.ylabel("Difference Norm")
plt.title("Norm of B_expected - B over Iterations")
plt.legend()

# Plot Loss over Iterations
plt.subplot(2, 2, 4)
plt.plot(loss_values, label="Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss over Iterations")
plt.legend()

plt.tight_layout()
plt.show()

