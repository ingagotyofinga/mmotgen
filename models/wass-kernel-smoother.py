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
import math
import random
from data.simulate_data import DataSimulator
from visualization.visualization import DataVisualizer

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
        return x


# this return makes sure the output has the right dimensions
# Selecting the second part of the output tensor

def box_kernel(mu_0, mu_i, bandwidth):
    # Compute Wasserstein-2 distance between mu_0 and mu_i
    # input_data_sample = mu_i[0].unsqueeze(0)
    # mu_0_sample = mu_0.unsqueeze(0)
    sink_loss = geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=0.05)
    wasserstein_distance = sink_loss(mu_0, mu_i)

    # Check if the Wasserstein-2 distance is less than or equal to h
    indicator = torch.tensor(1.0) if wasserstein_distance <= bandwidth else torch.tensor(0.0)

    # Compute the kernel
    kernel = (1 / (2 * bandwidth)) * indicator

    return kernel


# Custom loss function incorporating Wasserstein-2 distance and kernel smoother
def custom_loss(push, localdf, source, target, bw, blur=0.05, p=2 ):  # custom_loss(tpush, mu0, inputs, target_dists, step)
    # computes the loss locally
    # computes W2 distance between T_mu0#mu_i (tpush_i) and nu_i using sinkhorn divergence
    # blur param bw sinkhorn and kernel loss
    sink_loss = geomloss.SamplesLoss(loss='sinkhorn', p=p, blur=blur)
    wasserstein_distance = [sink_loss(push[i], target[i]) for i in
                            range(batch_size)]
    wasserstein_distance = torch.stack(wasserstein_distance)
    # Compute the kernel
    kernel = [box_kernel(localdf, source[i], bw) for i in range(batch_size)]
    kernel = torch.stack(kernel)
    # Combine the Wasserstein-2 distance and the kernel smoother term
    total_loss = torch.sum(
        wasserstein_distance * kernel)  # Combine your Wasserstein-2 loss and smoother term appropriately

    return total_loss


# TODO: new model validation script
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

# # SIMULATE DATA
# Number of distributions
num_distributions = 100
num_bins = 100
num_dimensions = 2

# means and sds for source data
means = np.random.randint(low=1,high=500, size=(num_distributions, num_dimensions))
std_devs = np.random.randint(low=1,high=50, size=(num_distributions, num_dimensions))

# generate data
simulator = DataSimulator(num_distributions, num_bins, num_dimensions)
source_dists = simulator.generate_source_data(means, std_devs)
target_dists = simulator.generate_target_data(source_dists)
input_data = simulator.generate_input_data(source_dists,target_dists)
mu0_distributions, step = simulator.generate_mu0_distributions(source_dists)


# Visualize the generated data
data = [source_dists, mu0_distributions]
labels = ["Source", "$\mu_0$"]
colors = ['blue', 'red']  # Color for each dataset

# Create an instance of DataVisualizer
visualizer = DataVisualizer(num_distributions, num_bins, num_dimensions)

# Visualize the data
visualizer.visualize(data, title="Randomly Generated Data", labels=labels, colors=colors)


# Compute mapping using univariate optimal transport map
# TODO: notebook to report results
# transport_map = univar_gaussian_transport_map(input_data[1][0], input_data[1][1], mu_source=None, sigma_source=None,
#                                               mu_target=None, sigma_target=None)
# sink_loss = geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=0.05)
# true_OTmap_loss = sink_loss(input_data[1][1].view(1, -1, 1), transport_map.view(1, -1, 1))

# Create TensorDataset and DataLoader for batching
source_dists_flat = source_dists.view(num_distributions, -1)
target_dists_flat = target_dists.view(num_distributions, -1)

# Split the data into training and test sets
source_train, source_test, target_train, target_test = train_test_split(
    source_dists_flat, target_dists_flat, test_size=0.2, random_state=42
)

# Create TensorDataset and DataLoader for training and test sets
train_dataset = TensorDataset(source_train, target_train)
test_dataset = TensorDataset(source_test, target_test)

batch_size = math.ceil(num_distributions*0.05)  # Set your desired batch size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# plt.plot(input_data[0][0], np.zeros_like(mu0), 'x')
# plt.plot(input_data[0][1], np.zeros_like(mu0), 'o')
# plt.show()

input_size = num_bins*num_dimensions # sample size of each distribution
hidden_size = 32  # Set your hidden size
output_size = num_bins*num_dimensions  # Set your output size
learning_rate = 0.01  # Set your learning rate

# model = OTMapNN(input_size, hidden_size, output_size)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# blur_values = [0.01, 0.05, 0.1]
blur_values = [0.01]
p_values = [1]

best_loss = float('inf')
best_params = {}

num_epochs = 10


for blur in blur_values:
    for p in p_values:
        print(f"Testing blur={blur}, p={p}")
        # Initialize lists to store losses
        all_train_losses = []
        all_test_losses = []
        all_predictions = []
        losses = []  # To store the loss values for each epoch

        # Dictionary to store models and optimizers
        models_and_optimizers = {}

        for idx, mu0_samples in enumerate(mu0_distributions):
            print(f'Training model for mu0_samples index: {idx}')
            mu0_tensor = mu0_samples.clone().detach().requires_grad_(False)
            model = OTMapNN(input_size, hidden_size, output_size)
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
            models_and_optimizers[idx] = (model, optimizer)

            train_losses = []
            for epoch in range(num_epochs):
                model.train()
                epoch_train_loss = []
                # losses_per_epoch = []  # to store average loss over all batches
                for source_batch, target_batch in train_loader:
                    source_batch = source_batch.clone().detach().requires_grad_(True)
                    source_batch = source_batch.view(batch_size, -1)
                    target_batch = target_batch.view(batch_size, num_bins, num_dimensions)

                    tpush = model(source_batch)
                    tpush = tpush.view(batch_size, num_bins, num_dimensions)
                    source_batch = source_batch.view(batch_size, num_bins, num_dimensions)

                    loss = custom_loss(tpush, mu0_tensor, source_batch, target_batch, step, blur, p)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # losses_per_epoch.append(loss.item())
                    epoch_train_loss.append(loss.item())

                train_loss = np.mean(epoch_train_loss)
                train_losses.append(train_loss)
                print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss}')

            all_train_losses.append(train_losses)

        # Evaluation Loop
        for idx, mu0_samples in enumerate(mu0_distributions):
            print(f'Evaluating model for mu0_samples index: {idx}')
            mu0_tensor = mu0_samples.clone().detach().requires_grad_(False)
            model, _ = models_and_optimizers[idx]
            model.eval()
            epoch_test_loss = []
            predictions = []

            with torch.no_grad():
                for source_batch, target_batch in test_loader:
                    source_batch = source_batch.view(batch_size, -1)
                    target_batch = target_batch.view(batch_size, num_bins, num_dimensions)

                    # Model forward pass
                    tpush = model(source_batch)
                    tpush = tpush.view(batch_size, num_bins, num_dimensions)
                    source_batch = source_batch.view(batch_size, num_bins, num_dimensions)
                    # Collect predictions
                    predictions.append(tpush)

                    # Compute loss
                    loss = custom_loss(tpush, mu0_tensor, source_batch, target_batch, step)

                    epoch_test_loss.append(loss.item())

            test_loss = np.mean(epoch_test_loss)
            all_test_losses.append(test_loss)
            all_predictions.append(torch.cat(predictions, dim=0))

            print(f'Test Loss for model {idx}: {test_loss}')

        # Combine predictions and reshape for visualization
        combined_predictions = torch.cat(all_predictions, dim=0)
        combined_predictions = combined_predictions.view(-1, num_bins, num_dimensions)

        avg_loss = np.mean(losses)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_params = {'blur': blur, 'p': p}

# TODO: notebook for results
# Plot the loss curves for each mu0_samples
for i, train_losses in enumerate(all_train_losses):
    plt.plot(train_losses, label=f'Train Loss {i}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train Loss Over Epochs')
plt.legend()
plt.show()

plt.figure()
plt.bar(range(len(all_test_losses)), all_test_losses)
plt.xlabel('Model Index')
plt.ylabel('Test Loss')
plt.title('Test Loss for Each Model')
plt.show()

# Plotting
# TODO: fix the plots with the updated model outputs
data = [source_dists, target_dists, combined_predictions.detach().numpy()]
labels = ["Source", "Target", "Predicted"]
colors = ['blue', 'green', 'red']  # Color for each dataset

# Create an instance of DataVisualizer
results = DataVisualizer(num_distributions, num_bins, num_dimensions)

# Visualize the data
results.visualize(data, title="Results", labels=labels, colors=colors)

# for idx in random_indices:
#     plt.scatter(tpush_numpy[idx], torch.zeros_like(tpush[0]), label='Predicted Target Distribution Samples', marker='+')
#     plt.scatter(input_data_numpy[idx][1], torch.zeros_like(input_data[0][1]), label='True Target Samples', marker='x')
#     transport_map = univar_gaussian_transport_map(input_data[idx][0], input_data[idx][1], mu_source=None,
#                                                   sigma_source=None,
#                                                   mu_target=None, sigma_target=None)
#     plt.scatter(transport_map, torch.zeros_like(transport_map), label='Optimally Mapped Target Samples', marker='o')
#     # Show the plot
#     plt.xlabel('Samples')
#     plt.ylabel('Distribution')
#     plt.title('Target Samples')
#     plt.legend()
#     plt.show()

# TODO: true_OTmap_loss & pred_OTmap_loss over all
# sink_loss = geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=0.05)
# true_OTmap_loss = sink_loss(input_data[1][1].view(1, -1, 1), transport_map.view(1, -1, 1))
# pred_OTmap_loss = sink_loss(input_data[1][1].view(1, -1, 1), tpush[1].view(1, -1, 1))
# print(f'True Loss: [{true_OTmap_loss}], Predicted Loss: {pred_OTmap_loss}')
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print or use the elapsed time as needed
print(f"Program execution time: {elapsed_time} seconds")
