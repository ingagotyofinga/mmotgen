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
# from visualization.visualization import DataVisualizer
# from multiprocessing import Pool, cpu_count
import pandas as pd

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


def univar_gaussian_transport_map(source_samples, target_samples, mu_source=None, sigma_source=None, mu_target=None,
                                  sigma_target=None):
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


def calculate_statistics(distributions):
    means = [torch.mean(dist) for dist in distributions]
    stds = [torch.std(dist) for dist in distributions]
    return means, stds


# SIMULATE DATA
num_distributions = 50
num_bins = 100
num_dimensions = 1

source_means = [[0] for _ in range(num_distributions)]
target_means = [[1] for _ in range(num_distributions)]
covariances = [[[1]] for _ in range(num_distributions)]
seed = 42  # Set a seed for reproducibility

simulator = DataSimulator(num_distributions, num_bins, source_means, target_means, covariances, seed)
source_means_std = 0.5  # Standard deviation for the noise added to source means
simulator.generate_data(source_means_std)
source_dists = simulator.data
# Generate target data
target_means_std = 0.5  # Standard deviation for the noise added to target means
target_dists = simulator.generate_target_data(source_means_std, target_means_std)
input_data = simulator.generate_input_data()
simulator.plot_data(simulator.data, 'Histogram of Generated 1D Gaussian Source Data')
simulator.plot_data(target_dists, 'Histogram of Generated 1D Gaussian Target Data')
mu0_distributions, step = simulator.generate_mu0_distributions()
simulator.plot_mu0_distributions(mu0_distributions, 'Histogram of Mu0 Distributions')

# Debug print for shapes after data generation
print("source_dists shape:", source_dists.shape)
print("target_dists shape:", target_dists.shape)
print("mu0_distributions shape:", mu0_distributions.shape)

source_dists_flat = source_dists.view(num_distributions, -1)
target_dists_flat = target_dists.view(num_distributions, -1)
source_train, source_test, target_train, target_test = train_test_split(source_dists_flat, target_dists_flat,
                                                                        test_size=0.2, random_state=42)

train_dataset = TensorDataset(source_train, target_train)
test_dataset = TensorDataset(source_test, target_test)

batch_size = math.ceil(num_distributions * 0.1)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

input_size = num_bins * num_dimensions
hidden_size = 32
output_size = num_bins * num_dimensions
learning_rate = 0.001

blur_values = [0.01, 0.025, 0.05, 0.075, 0.1]

best_loss = float('inf')
best_params = {}
num_epochs = 50  # Initial reasonable number of epochs
patience = 5  # Early stopping patience


def train_model(args):
    idx, blur = args
    mu0_samples = mu0_distributions[idx]
    mu0_tensor = mu0_samples.clone().detach().requires_grad_(False)
    model = OTMapNN(input_size, hidden_size, output_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Changed to Adam optimizer

    train_losses = []
    test_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = []

        for source_batch, target_batch in train_loader:

            source_batch = source_batch.clone().detach().requires_grad_(True)
            source_batch = source_batch.view(batch_size, -1)
            target_batch = target_batch.view(batch_size, -1)

            tpush = model(source_batch)
            tpush = tpush.view(batch_size, num_bins, num_dimensions)
            source_batch = source_batch.view(batch_size, num_bins, num_dimensions)
            target_batch = target_batch.view(batch_size, num_bins, num_dimensions)

            loss = custom_loss(tpush, mu0_tensor, source_batch, target_batch, step, blur=blur, batch_size=batch_size)

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
                target_batch = target_batch.view(batch_size, -1)

                tpush = model(source_batch)
                tpush = tpush.view(batch_size, num_bins, num_dimensions)
                source_batch = source_batch.view(batch_size, num_bins, num_dimensions)
                target_batch = target_batch.view(batch_size, num_bins, num_dimensions)

                loss = custom_loss(tpush, mu0_tensor, source_batch, target_batch, step, blur=blur, batch_size=batch_size)
                val_loss.append(loss.item())

        avg_val_loss = np.mean(val_loss)
        test_losses.append(avg_val_loss)
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

    test_means = []
    test_stds = []
    predicted_means = []
    predicted_stds = []

    with torch.no_grad():
        for source_batch, target_batch in test_loader:
            source_batch = source_batch.view(batch_size, -1)
            target_batch = target_batch.view(batch_size, -1)

            tpush = model(source_batch)
            tpush = tpush.view(batch_size, num_bins, num_dimensions)
            source_batch = source_batch.view(batch_size, num_bins, num_dimensions)
            target_batch = target_batch.view(batch_size, num_bins, num_dimensions)
            predictions.append(tpush)

            loss = custom_loss(tpush, mu0_tensor, source_batch, target_batch, step, blur=blur, batch_size=batch_size)
            epoch_test_loss.append(loss.item())

            # Calculate sample mean and standard deviation for each distribution in the batch
            for tb, pb in zip(target_batch, tpush):
                test_means.append(tb.mean().item())
                test_stds.append(tb.std().item())
                predicted_means.append(pb.mean().item())
                predicted_stds.append(pb.std().item())

    test_loss = np.mean(epoch_test_loss)
    predictions = torch.cat(predictions, dim=0)

    print(f'Test Loss for model {idx} with blur={blur}: {test_loss}')

    return train_losses, test_losses, test_loss, predictions, test_means, test_stds, predicted_means, predicted_stds


if __name__ == "__main__":
    all_train_losses = []
    all_test_losses = []
    final_test_losses = []
    all_predictions = []
    results_list = []

    for blur in blur_values:
        print(f"Testing blur={blur}")

        # Initialize lists to collect results for the current blur value
        train_losses_blur = []
        test_losses_blur = []
        final_test_losses_blur = []
        predictions_blur = []

        # Use single process
        results = [train_model((i, blur)) for i in range(len(mu0_distributions))]
        # Uncomment the following lines to use multi-processing if needed
        # with Pool(processes=cpu_count()) as pool:
        #     results = pool.map(train_model, [(i, blur) for i in range(len(mu0_distributions))])

        for train_losses, test_losses, test_loss, predictions, test_means, test_stds, predicted_means, predicted_stds in results:
            train_losses_blur.append(train_losses)
            test_losses_blur.append(test_losses)
            final_test_losses_blur.append(test_loss)
            predictions_blur.append(predictions)

            # Store the results for each test distribution
            for tm, ts, pm, ps in zip(test_means, test_stds, predicted_means, predicted_stds):
                relative_error_mean = abs(tm - pm) / abs(tm) if tm != 0 else float('inf')
                relative_error_std = abs(ts - ps) / abs(ts) if ts != 0 else float('inf')
                results_list.append({
                    'blur': blur,
                    'test_mean': tm,
                    'test_std': ts,
                    'predicted_mean': pm,
                    'predicted_std': ps,
                    'relative_error_mean': relative_error_mean,
                    'relative_error_std': relative_error_std
                })

        combined_predictions = torch.cat(predictions_blur, dim=0)
        combined_predictions = combined_predictions.view(-1, num_bins, num_dimensions)

        avg_loss = np.mean(final_test_losses_blur)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_params = {'blur': blur}

        # Padding the training and test loss sequences
        max_train_length = max(len(train_loss) for train_loss in train_losses_blur)
        padded_train_losses = [
            np.pad(train_loss, (0, max_train_length - len(train_loss)), 'constant', constant_values=np.nan) for
            train_loss in train_losses_blur]

        max_test_length = max(len(test_loss) for test_loss in test_losses_blur)
        padded_test_losses = [
            np.pad(test_loss, (0, max_test_length - len(test_loss)), 'constant', constant_values=np.nan) for test_loss
            in test_losses_blur]

        # Convert to numpy arrays for easy manipulation
        train_losses_array = np.array(padded_train_losses)
        test_losses_array = np.array(padded_test_losses)

        # Calculate mean and standard deviation across models for each epoch
        mean_train_loss = np.nanmean(train_losses_array, axis=0)
        std_train_loss = np.nanstd(train_losses_array, axis=0)

        mean_test_loss = np.nanmean(test_losses_array, axis=0)
        std_test_loss = np.nanstd(test_losses_array, axis=0)

        # Plot mean training and test loss with shaded standard deviation
        epochs = range(len(mean_train_loss))
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, mean_train_loss, label='Mean Train Loss')
        plt.fill_between(epochs, mean_train_loss - std_train_loss, mean_train_loss + std_train_loss, alpha=0.2)
        plt.plot(epochs, mean_test_loss, label='Mean Test Loss')
        plt.fill_between(epochs, mean_test_loss - std_test_loss, mean_test_loss + std_test_loss, alpha=0.2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Mean Train and Test Loss Over Epochs with Standard Deviation (blur={blur})')
        plt.legend()
        plt.show()

        all_train_losses.append(train_losses_blur)
        all_test_losses.append(test_losses_blur)
        final_test_losses.append(final_test_losses_blur)
        all_predictions.append(predictions_blur)

        # Select indices of best, median, and worst-performing models
        final_losses = [loss[-1] if loss else float('inf') for loss in train_losses_blur]
        best_model_idx = np.argmin(final_losses)
        worst_model_idx = np.argmax(final_losses)
        median_model_idx = np.argsort(final_losses)[len(final_losses) // 2]

        # Plot the losses for these models
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses_blur[best_model_idx], label='Best Model Train Loss')
        plt.plot(test_losses_blur[best_model_idx], label='Best Model Test Loss')
        plt.plot(train_losses_blur[median_model_idx], label='Median Model Train Loss')
        plt.plot(test_losses_blur[median_model_idx], label='Median Model Test Loss')
        plt.plot(train_losses_blur[worst_model_idx], label='Worst Model Train Loss')
        plt.plot(test_losses_blur[worst_model_idx], label='Worst Model Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Train and Test Loss Over Epochs for Best, Median, and Worst Models (blur={blur})')
        plt.legend()
        plt.show()

        # Plot final test loss for each model
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(final_test_losses_blur)), final_test_losses_blur)
        plt.xlabel('Model Index')
        plt.ylabel('Test Loss')
        plt.title(f'Final Test Loss for Each Model (blur={blur})')
        plt.show()

        # Select a subset of distributions to plot
        num_samples_to_plot = 2
        sample_indices = random.sample(range(len(test_dataset)), num_samples_to_plot)

        # Prepare data for visualization
        selected_target_dists = target_dists[sample_indices]
        selected_predictions = combined_predictions[sample_indices].detach().numpy()

        # Define colors for each pair of distributions
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        light_colors = ['#ff9999', '#99ccff', '#99ff99', '#ffcc99', '#cc99ff']
        data = [selected_target_dists, selected_predictions]

        plt.figure(figsize=(12, 6))
        for i in range(num_samples_to_plot):
            plt.hist(selected_target_dists[i][:, 0], bins=10, alpha=0.5, color=colors[i], edgecolor='black',
                     label=f'Target {sample_indices[i]}')
            plt.hist(selected_predictions[i][:, 0], bins=10, alpha=0.5, color=light_colors[i], edgecolor='black',
                     label=f'Predicted {sample_indices[i]}')
        plt.title(f'Comparison of Selected True and Predicted Distributions (1d Gaussian) (blur={blur})')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Convert results_list to a DataFrame
    results_df = pd.DataFrame(results_list)
    print(results_df)

    # Save the DataFrame to a CSV file for further analysis
    results_df.to_csv('results.csv', index=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Program execution time: {elapsed_time} seconds")
