import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np
import geomloss
import matplotlib.pyplot as plt
import time
import random
from ignite.engine import Engine, Events, State
from ignite.handlers import EarlyStopping

start_time = time.time()
# Set a seed for reproducibility
torch.manual_seed(42)


class OTMapNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(OTMapNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x[:, 1, :]


# this return makes sure the output has the right dimensions
# Selecting the second part of the output tensor

def box_kernel(mu_0, mu_i, bandwidth):
    # Compute Wasserstein-2 distance between mu_0 and mu_i
    input_data_sample = mu_i[0].unsqueeze(0)
    mu_0_sample = mu_0.unsqueeze(0)
    sink_loss = geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=0.05)
    wasserstein_distance = sink_loss(mu_0_sample, input_data_sample)

    # Check if the Wasserstein-2 distance is less than or equal to h
    indicator = torch.tensor(1.0) if wasserstein_distance <= bandwidth else torch.tensor(0.0)

    # Compute the kernel
    kernel = (1 / (2 * bandwidth)) * indicator

    return kernel


# Custom loss function incorporating Wasserstein-2 distance and kernel smoother
def custom_loss(push, localdf, dists, bw):  # custom_loss(tpush, mu0, inputs, step)
    # computes the loss locally
    # computes W2 distance between T_mu0#mu_i (tpush_i) and nu_i using sinkhorn divergence
    # blur param bw sinkhorn and kernel loss
    sink_loss = geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=0.05)
    wasserstein_distance = [sink_loss(push[i][1].view(1, -1, 1), dists[i][1].view(1, -1, 1)) for i in
                            range(batch_size)]
    wasserstein_distance = torch.stack(wasserstein_distance)
    # Compute the kernel
    kernel = [box_kernel(localdf, dists[i], bw) for i in range(batch_size)]
    kernel = torch.stack(kernel)
    # Combine the Wasserstein-2 distance and the kernel smoother term
    total_loss = torch.sum(
        wasserstein_distance * kernel)  # Combine your Wasserstein-2 loss and smoother term appropriately

    return total_loss


def univar_gaussian_transport_map(source_samples, target_samples, mu_source=None, sigma_source=None, mu_target=None,
                                  sigma_target=None):
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
        mu_source = torch.mean(source_samples)
    if sigma_source is None:
        sigma_source = torch.std(source_samples)
    if mu_target is None:
        mu_target = torch.mean(target_samples)
    if sigma_target is None:
        sigma_target = torch.std(target_samples)
    # You can similarly handle mu_target and sigma_target using sample estimators

    mapped_samples = mu_target + (sigma_target / sigma_source) * (source_samples - mu_source)
    return mapped_samples


# Define the training function
def train_step(engine, batch):
    model, optimizer, criterion, mu0_tensor, step = engine.state.model, engine.state.optimizer, engine.state.criterion, engine.state.mu0_tensor, engine.state.step
    model.train()  # Set model to training mode
    # Unpack batch data
    inputs, targets = batch
    # Forward pass
    tpush = model(inputs)
    # Calculate loss
    loss = criterion(tpush, mu0_tensor, inputs, step * 100)
    # Clear gradients
    optimizer.zero_grad()
    # Backpropagation
    loss.backward()
    # Parameter update
    optimizer.step()
    return loss.item()  # Return loss value


# Define the evaluation function
def evaluate_step(engine, batch):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        inputs = batch[0]  # Unpack batch
        outputs = model(inputs)  # Forward pass
        return outputs


# SIMULATE DATA
# Number of distributions
num_distributions = 100
num_bins = 100

# mu0_mean = torch.randn(1).item()
# mu0_std = torch.abs(torch.randn(1)).item()
# mu0 = torch.normal(mean=mu0_mean, std=mu0_std, size=(num_bins,))
# mu0_samples = torch.abs(mu0_samples)
# mu0 = mu0_samples / mu0_samples.sum()

# plt.plot(mu0, np.zeros_like(mu0), 'x')

# Lists to store source and target distributions
mu_distributions = []
nu_distributions = []

for _ in range(num_distributions):
    mu_mean = torch.randn(1).item()
    mu_std = torch.abs(torch.randn(1)).item()
    mu_samples = torch.normal(mean=mu_mean, std=mu_std, size=(num_bins,))
    # mu_samples = torch.abs(mu_samples)
    # mu_distribution = mu_samples / mu_samples.sum()
    mu_distributions.append(mu_samples)
    # plt.plot(mu_samples, np.zeros_like(mu0), 'x')

    nu_mean = torch.randn(1).item()
    nu_std = torch.abs(torch.randn(1)).item()
    nu_samples = torch.normal(mean=nu_mean, std=nu_std, size=(num_bins,))
    # nu_samples = torch.abs(nu_samples)
    # nu_distribution = nu_samples / nu_samples.sum()
    nu_distributions.append(nu_samples)
    # plt.plot(nu_samples, np.zeros_like(mu0), 'x')
source_dists = torch.stack(mu_distributions)
target_dists = torch.stack(nu_distributions)
input_data = torch.cat([torch.stack(mu_distributions).unsqueeze(1), torch.stack(nu_distributions).unsqueeze(1)], dim=1)
input_data = input_data.view(num_distributions, -1, num_bins)

# plt.title('Mu Distributions')
# plt.show()

source_min = torch.min(source_dists)
source_range = np.abs(torch.max(source_dists) - source_min)
num_mu0s = int(np.ceil(2 * source_range))
step = source_range / num_mu0s  # this is going to act as the bandwidth for now (h)
mu0_means = [source_min + k * step for k in range(int(num_mu0s))]
mu0_std = torch.abs(step)

mu0_distributions = []
for i in range(num_mu0s):
    mu0_samples = torch.normal(mean=mu0_means[i], std=mu0_std, size=(num_bins,))
    mu0_distributions.append(mu0_samples)
    plt.plot(mu0_samples, np.zeros_like(mu0_samples), 'x')

plt.show()

# Compute mapping using univariate optimal transport map
transport_map = univar_gaussian_transport_map(input_data[1][0], input_data[1][1], mu_source=None, sigma_source=None,
                                              mu_target=None, sigma_target=None)
sink_loss = geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=0.05)
true_OTmap_loss = sink_loss(input_data[1][1].view(1, -1, 1), transport_map.view(1, -1, 1))

# Create TensorDataset and DataLoader for batching
dataset = TensorDataset(input_data)
batch_size = 100  # Set your desired batch size
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# plt.plot(input_data[0][0], np.zeros_like(mu0), 'x')
# plt.plot(input_data[0][1], np.zeros_like(mu0), 'o')
# plt.show()

input_size = num_bins  # sample size of each distribution
hidden_size = 32  # Set your hidden size
output_size = num_bins  # Set your output size
learning_rate = 0.002  # Set your learning rate

model = OTMapNN(input_size, hidden_size, output_size)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
trainer = Engine(train_step)
trainer.state.model = model
trainer.state.optimizer = optimizer
trainer.state.criterion = custom_loss
trainer.state.step = step
evaluator = Engine(evaluate_step)


# Define event handlers
@trainer.on(Events.EPOCH_COMPLETED)
def log_training_loss(engine):
    print(f"Epoch {engine.state.epoch}/{num_epochs}, Loss: {engine.state.output:.6f}")


num_epochs = 1000
losses = []  # To store the loss values for each epoch


# Define score function for early stopping based on training loss
def score_function(engine):
    epoch_loss = engine.state.metrics['epoch_loss']
    return -epoch_loss


early_stopping = EarlyStopping(patience=5, score_function=lambda engine: -engine.state.metrics['validation_loss'],
                               trainer=trainer)
evaluator.add_event_handler(Events.COMPLETED, early_stopping)

# TODO: Why is the loss 0.? Happened after adding more than one mu0_distributions.
for mu0_samples in mu0_distributions:
    tpush_list = []  # Initialize a list to store tpush for the current mu0
    mu0_tensor = mu0_samples.clone().detach().requires_grad_(False)
    trainer.state.mu0_tensor = mu0_tensor
    for epoch in range(num_epochs):
        losses_per_epoch = []  # to store average loss over all batches
        for batch_data in dataloader:
            loss = train_step(model, optimizer, custom_loss, batch_data, mu0_tensor, step)
            losses_per_epoch.append(loss)

        epoch_loss = np.mean(losses_per_epoch)
        losses.append(epoch_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss}')

        # Check if early stopping criteria met
        if early_stopping(engine):
            print("Early stopping triggered.")
            break

    # Store tpush for each mu0
    tpush_list.append(tpush.detach().numpy())

trainer.run(dataloader, max_epochs=num_epochs)
# evaluator.run(val_loader) need to add val_loader once I have a test set.

# Plot the loss curve
plt.plot(losses, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.legend()
plt.show()

# Randomly select ten indices
random_indices = random.sample(range(num_distributions), 10)

# Convert tensors to NumPy arrays for plotting
tpush_numpy = tpush.detach().numpy()
input_data_numpy = input_data.detach().numpy()

# Plotting
for idx in random_indices:
    plt.scatter(tpush_numpy[idx], torch.zeros_like(tpush[0]), label='Predicted Target Distribution Samples', marker='+')
    plt.scatter(input_data_numpy[idx][1], torch.zeros_like(input_data[0][1]), label='True Target Samples', marker='x')
    transport_map = univar_gaussian_transport_map(input_data[idx][0], input_data[idx][1], mu_source=None,
                                                  sigma_source=None,
                                                  mu_target=None, sigma_target=None)
    plt.scatter(transport_map, torch.zeros_like(transport_map), label='Optimally Mapped Target Samples', marker='o')
    # Show the plot
    plt.xlabel('Samples')
    plt.ylabel('Distribution')
    plt.title('Target Samples')
    plt.legend()
    plt.show()

# TODO: true_OTmap_loss & pred_OTmap_loss over all
sink_loss = geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=0.05)
true_OTmap_loss = sink_loss(input_data[1][1].view(1, -1, 1), transport_map.view(1, -1, 1))
pred_OTmap_loss = sink_loss(input_data[1][1].view(1, -1, 1), tpush[1].view(1, -1, 1))
print(f'True Loss: [{true_OTmap_loss}], Predicted Loss: {pred_OTmap_loss}')
# end_time = time.time()
#
# # Calculate the elapsed time
# elapsed_time = end_time - start_time
#
# # Print or use the elapsed time as needed
# print(f"Program execution time: {elapsed_time} seconds")
