import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import geomloss
import matplotlib.pyplot as plt
import time
import math
import random
from sklearn.model_selection import train_test_split
from data.simulate_data import DataSimulator
from visualization.visualization import DataVisualizer
from multiprocessing import Pool, cpu_count

start_time = time.time()
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

def box_kernel(mu_0, mu_i, bandwidth):
    sink_loss = geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=0.05)
    wasserstein_distance = sink_loss(mu_0, mu_i)
    indicator = torch.tensor(1.0) if wasserstein_distance <= bandwidth else torch.tensor(0.0)
    kernel = (1 / (2 * bandwidth)) * indicator
    return kernel

def custom_loss(push, localdf, source, target, bw, blur=0.05, p=2, batch_size=1):
    sink_loss = geomloss.SamplesLoss(loss='sinkhorn', p=p, blur=blur)
    wasserstein_distance = [sink_loss(push[i], target[i]) for i in range(batch_size)]
    wasserstein_distance = torch.stack(wasserstein_distance)
    kernel = [box_kernel(localdf, source[i], bw) for i in range(batch_size)]
    kernel = torch.stack(kernel)
    total_loss = torch.sum(wasserstein_distance * kernel)
    return total_loss

def univar_gaussian_transport_map(source_samples, target_samples, mu_source=None, sigma_source=None, mu_target=None, sigma_target=None):
    if mu_source is None:
        mu_source = torch.mean(source_samples)
    if sigma_source is None:
        sigma_source = torch.std(source_samples)
    if mu_target is None:
        mu_target = torch.mean(target_samples)
    if sigma_target is None:
        sigma_target = torch.std(target_samples)
    mapped_samples = mu_target + (sigma_target / sigma_source) * (source_samples - mu_source)
    return mapped_samples

# SIMULATE DATA
num_distributions = 100
num_bins = 100
num_dimensions = 2

means = np.random.randint(low=1, high=500, size=(num_distributions, num_dimensions))
std_devs = np.random.randint(low=1, high=50, size=(num_distributions, num_dimensions))

simulator = DataSimulator(num_distributions, num_bins, num_dimensions)
source_dists = simulator.generate_source_data(means, std_devs)
target_dists = simulator.generate_target_data(source_dists)
input_data = simulator.generate_input_data(source_dists, target_dists)
mu0_distributions, step = simulator.generate_mu0_distributions(source_dists)

# Visualize the generated data
data = [source_dists, mu0_distributions]
labels = ["Source", "$\mu_0$"]
colors = ['blue', 'red']
visualizer = DataVisualizer(num_distributions, num_bins, num_dimensions)
visualizer.visualize(data, title="Randomly Generated Data", labels=labels, colors=colors)

source_dists_flat = source_dists.view(num_distributions, -1)
target_dists_flat = target_dists.view(num_distributions, -1)
source_train, source_test, target_train, target_test = train_test_split(source_dists_flat, target_dists_flat, test_size=0.2, random_state=42)

train_dataset = TensorDataset(source_train, target_train)
test_dataset = TensorDataset(source_test, target_test)

batch_size = math.ceil(num_distributions * 0.05)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

input_size = num_bins * num_dimensions
hidden_size = 32
output_size = num_bins * num_dimensions
learning_rate = 0.01

blur_values = [0.01]
p_values = [1]

best_loss = float('inf')
best_params = {}
num_epochs = 50  # Initial reasonable number of epochs
patience = 10  # Early stopping patience

def train_model(args):
    idx, blur, p = args
    mu0_samples = mu0_distributions[idx]
    mu0_tensor = mu0_samples.clone().detach().requires_grad_(False)
    model = OTMapNN(input_size, hidden_size, output_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Changed to Adam optimizer

    train_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = []

        for source_batch, target_batch in train_loader:
            source_batch = source_batch.clone().detach().requires_grad_(True)
            source_batch = source_batch.view(batch_size, -1)
            target_batch = target_batch.view(batch_size, num_bins, num_dimensions)

            tpush = model(source_batch)
            tpush = tpush.view(batch_size, num_bins, num_dimensions)
            source_batch = source_batch.view(batch_size, num_bins, num_dimensions)

            loss = custom_loss(tpush, mu0_tensor, source_batch, target_batch, step, blur, p, batch_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss.append(loss.item())

        train_loss = np.mean(epoch_train_loss)
        train_losses.append(train_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss}')

        # Early stopping check
        model.eval()
        val_loss = []
        with torch.no_grad():
            for source_batch, target_batch in test_loader:
                source_batch = source_batch.view(batch_size, -1)
                target_batch = target_batch.view(batch_size, num_bins, num_dimensions)

                tpush = model(source_batch)
                tpush = tpush.view(batch_size, num_bins, num_dimensions)
                source_batch = source_batch.view(batch_size, num_bins, num_dimensions)

                loss = custom_loss(tpush, mu0_tensor, source_batch, target_batch, step, blur, p, batch_size)
                val_loss.append(loss.item())

        avg_val_loss = np.mean(val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Evaluate model
    model.eval()
    epoch_test_loss = []
    predictions = []

    with torch.no_grad():
        for source_batch, target_batch in test_loader:
            source_batch = source_batch.view(batch_size, -1)
            target_batch = target_batch.view(batch_size, num_bins, num_dimensions)

            tpush = model(source_batch)
            tpush = tpush.view(batch_size, num_bins, num_dimensions)
            source_batch = source_batch.view(batch_size, num_bins, num_dimensions)
            predictions.append(tpush)

            loss = custom_loss(tpush, mu0_tensor, source_batch, target_batch, step, blur, p, batch_size)
            epoch_test_loss.append(loss.item())

    test_loss = np.mean(epoch_test_loss)
    predictions = torch.cat(predictions, dim=0)

    print(f'Test Loss for model {idx}: {test_loss}')

    return train_losses, test_loss, predictions

if __name__ == "__main__":
    for blur in blur_values:
        for p in p_values:
            print(f"Testing blur={blur}, p={p}")
            all_train_losses = []
            all_test_losses = []
            all_predictions = []
            losses = []

            # with Pool(processes=cpu_count()) as pool:
            #     results = pool.map(train_model, [(i, blur, p) for i in range(len(mu0_distributions))])
            results = [train_model(i, blur, p) for i in range(len(mu0_distributions))]  # Single process

        for train_losses, test_loss, predictions in results:
                all_train_losses.append(train_losses)
                all_test_losses.append(test_loss)
                all_predictions.append(predictions)

            combined_predictions = torch.cat(all_predictions, dim=0)
            combined_predictions = combined_predictions.view(-1, num_bins, num_dimensions)

            avg_loss = np.mean(losses)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_params = {'blur': blur, 'p': p}

    for i, train_losses in enumerate(all_train_losses):
        plt.plot(train_losses)
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

    data = [target_dists, combined_predictions.detach().numpy()]
    labels = ["Target", "Predicted"]
    colors = ['green', 'red']

    results = DataVisualizer(num_distributions, num_bins, num_dimensions)
    results.visualize(data, title="Results", labels=labels, colors=colors)

    # Select a subset of distributions to plot
    num_samples_to_plot = 5
    sample_indices = random.sample(range(num_distributions), num_samples_to_plot)

    # Prepare data for visualization
    selected_target_dists = target_dists[sample_indices]
    selected_predictions = combined_predictions[sample_indices].detach().numpy()

    # Define colors for each pair of distributions
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    light_colors = ['#ff9999', '#99ccff', '#99ff99', '#ffcc99', '#cc99ff']
    data = [selected_target_dists, selected_predictions]

    # Plot the selected distributions
    plt.figure(figsize=(12, 8))
    for i in range(num_samples_to_plot):
        plt.scatter(selected_target_dists[i][:, 0], selected_target_dists[i][:, 1], color=colors[i], label=f'Target {sample_indices[i]}')
        plt.scatter(selected_predictions[i][:, 0], selected_predictions[i][:, 1], color=light_colors[i], label=f'Predicted {sample_indices[i]}', marker='x')

    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Comparison of Selected True and Predicted Distributions (2D Point Clouds)')
    plt.legend()
    plt.show()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Program execution time: {elapsed_time} seconds")
