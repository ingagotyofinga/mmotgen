import torch
import torch.nn as nn
import torch.optim as optim
import geomloss
import matplotlib.pyplot as plt

class OTMapNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(OTMapNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def box_kernel(mu_0, mu_i, h):
    # Compute Wasserstein-2 distance between mu_0 and mu_i
    sink_loss = geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=0.0)
    wasserstein_distance = sink_loss(mu_0.view(1, -1, 1), mu_i.view(1, -1, 1))

    # Check if the Wasserstein-2 distance is less than or equal to h
    indicator = torch.tensor(1.0) if wasserstein_distance <= h else torch.tensor(0.0)

    # Compute the kernel
    kernel = (1 / (2 * h)) * indicator

    return kernel

# Custom loss function incorporating Wasserstein-2 distance and kernel smoother
def custom_loss(Tpush_i, mu_0, mu_i, nu_i):
    # computes the loss locally
    # computes W2 distance between T_mu0#mu_i (Tpush_i) and nu_i using sinkhorn divergence
    # blur param bw sinkhorn and kernel loss
    sink_loss = geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=0.0)
    wasserstein_distance = sink_loss(Tpush_i.view(1, -1, 1), nu_i.view(1, -1, 1))

    # Compute the kernel
    Kh = box_kernel(mu_0, mu_i, h)

    # Combine the Wasserstein-2 distance and the kernel smoother term
    total_loss = wasserstein_distance*Kh # Combine your Wasserstein-2 loss and smoother term appropriately

    return total_loss

# Set seed for reproducibility
torch.manual_seed(42)

# Number of distributions
num_distributions = 100
num_bins = 100

# Lists to store mu distributions
mu_distributions = []
nu_distributions = []

for _ in range(num_distributions):
    mu_mean = torch.randn(1).item()
    mu_std = torch.abs(torch.randn(1)).item()
    mu_samples = torch.normal(mean=mu_mean, std=mu_std, size=(num_bins,))
    mu_samples = torch.abs(mu_samples)
    mu_distribution = mu_samples / mu_samples.sum()
    mu_distributions.append(mu_distribution)

    nu_mean = torch.randn(1).item()
    nu_std = torch.abs(torch.randn(1)).item()
    nu_samples = torch.normal(mean=nu_mean, std=nu_std, size=(num_bins,))
    nu_samples = torch.abs(nu_samples)
    nu_distribution = nu_samples / nu_samples.sum()
    nu_distributions.append(nu_distribution)

source_dists = torch.stack(mu_distributions)
target_dists = torch.stack(nu_distributions)

# Example usage in training loop
input_size = num_bins  # sample size of each distribution
hidden_size = 64 # Set your hidden size
output_size = 3  # Set your output size

model = OTMapNN(input_size, hidden_size, output_size)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

num_epochs = 100

for epoch in range(num_epochs):
    # Assuming inputs, mu_i, nu_i, and Kh are your data and kernel smoother parameters
    inputs = torch.tensor(input_data, requires_grad=True)
    mu_i = torch.tensor(mu_samples, requires_grad=False)
    nu_i = torch.tensor(nu_samples, requires_grad=False)
    Kh = torch.tensor(Kh_data, requires_grad=False)

    # Forward pass
    T_mu0 = model(inputs)

    # Compute loss
    loss = sum(custom_loss(Tpush[i], mu0[i], mu[i], nu[i]) for i in range(input_size))


    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')
