# Univariate Gaussian Piecewise Smooth Map Neural Wasserstein Regression

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.animation import FuncAnimation
from sklearn.model_selection import train_test_split
import os
from datetime import datetime

# Get the current date
current_date = datetime.now().strftime("%Y-%m-%d")

# Seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Data Generation
def generate_gaussian_data(num_samples, mean, std, num_distributions):
    data = []
    half_distributions = num_distributions // 2  # Get half the number of distributions
    mean1 = mean + 1
    mean2 = mean + 15
    # Generate the first half of the distributions with mean1
    for _ in range(half_distributions):
        samples = torch.distributions.MultivariateNormal(mean1, std).sample((num_samples,))
        data.append(samples)

    # Generate the second half of the distributions with mean2
    for _ in range(num_distributions - half_distributions):  # In case num_distributions is odd
        samples = torch.distributions.MultivariateNormal(mean2, std).sample((num_samples,))
        data.append(samples)

    return torch.stack(data)

def generate_refmeasure(data, cov, num_samples):
    # Calculate min and max over all elements
    data_min = torch.min(data)
    data_max = torch.max(data)

    # Calculate range
    data_range = torch.abs(data_max - data_min)

    # Define the number of reference measures and calculate step size
    num_mu0s = 2  # You can adjust this based on your requirements
    step = data_range / (num_mu0s+1)

    # Generate evenly spaced means for reference measures
    mu0_means = torch.linspace(data_min + step, data_min + (num_mu0s) * step, num_mu0s).unsqueeze(1)  # Ensure it's 1D

    # Generate samples for each reference measure (assuming same covariance for all)
    mu0 = torch.stack([torch.distributions.MultivariateNormal(mean, cov).sample((num_samples,))
                       .squeeze(-1)  # Remove the extra dimension
                       for mean in mu0_means])

    return mu0

# Univariate Gaussian Wasserstein Distance p=1
def earthmover(data1, data2):
    mean1 = data1.mean(dim=0)
    mean2 = data2.mean(dim=0)
    cov1 = data1.cov()
    cov2 = data2.cov()

    return torch.abs(mean1 - mean2) + torch.abs(cov1 - cov2)

# Univariate Gaussian Wasserstein Distance p=2
def w2(data1, data2):
    mean1 = data1.mean(dim=0)
    mean2 = data2.mean(dim=0)
    cov1 = data1.var()
    cov2 = data2.var()

    return torch.sqrt((mean1 - mean2)**2 + (cov1 - cov2)**2)

def sigmoid(W, theta=1, delta=8):
    """
    Sigmoid function based on Wasserstein distance.

    Parameters:
    W (float): The Wasserstein distance.
    theta (float): Parameter controlling the steepness of the sigmoid function.
    epsilon (float): Threshold for the Wasserstein distance.

    Returns:
    float: The value of the sigmoid function.
    """
    return 1 / (1 + torch.exp(-theta * (W - delta)))

# For data simulation
# piecewise smooth map
def compute_target_data(source_data, bm_data):
    num_sources = len(source_data)

    target_data = []

    #loop over source_data and
    for i in range(num_sources):
        w2_dist = w2(source_data[i], bm_data)  # W2 distance between source[i] and ref[j]
        sig = sigmoid(w2_dist)  # Sigmoid of W2 distance

        # Compute the target_data for this pair (i, j)
        target_value = (source_data[i] - 12) * sig + (-1* source_data[i] + 4) * (1 - sig)
        target_data.append(target_value)

    # Convert the list to a torch tensor
    target_data_tensor = torch.stack(target_data)
    # target_data_tensor = target_data_tensor + torch.randn_like(target_data_tensor)

    return target_data_tensor


num_samples = 1000
num_distributions = 1000
num_dimensions = 1
source_mean = torch.zeros(num_dimensions)
source_cov = torch.eye(num_dimensions)
# target_mean = 7  # Shifted to 7 to be significantly different from source
# target_std = 2

source_data = generate_gaussian_data(num_samples, source_mean, source_cov, num_distributions)
ref_data = generate_refmeasure(source_data, source_cov, num_samples)
bm_data = torch.distributions.MultivariateNormal(source_mean, source_cov).sample((num_samples,))
target_data = compute_target_data(source_data, bm_data)

# Flatten data for training
source_data_flat = source_data.reshape(num_distributions, -1)
target_data_flat = target_data.reshape(num_distributions, -1)

# Train-test split
source_train, source_test, target_train, target_test = train_test_split(source_data_flat, target_data_flat,
                                                                        test_size=0.2, random_state=42)

train_dataset = TensorDataset(torch.Tensor(source_train), torch.Tensor(target_train))
test_dataset = TensorDataset(torch.Tensor(source_test), torch.Tensor(target_test))

batch_size = 1  # Adjusted batch size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model Definition
#class SimpleNN(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(SimpleNN, self).__init__()
#         self.fc1 = nn.Linear(input_dim, input_dim)
#         self.fc2 = nn.Linear(input_dim, output_dim)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.fc2(x)
#         return x

class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()  # Adding ReLU activation for non-linearity
        self.dropout = nn.Dropout(0.3)  # Adding dropout to prevent overfitting

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)  # No activation for output layer
        return x


input_dim = num_samples
output_dim = num_samples

model = SimpleNN(input_dim, output_dim)

# Initialize weights
def initialize_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)

model.apply(initialize_weights)

# Print initial weights and biases
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name} initial values:\n{param.data}\n")


def gaussian_w2(ref, source, pred, target, bandwidth=1):
    # Compute the means and stds for pred, source, and target
    pred_mean = torch.mean(pred, dim=1)
    pred_std = torch.std(pred, dim=1)

    source_mean = torch.mean(source, dim=1)
    source_std = torch.std(source, dim=1)

    target_mean = torch.mean(target, dim=1)
    target_std = torch.std(target, dim=1)

    # Univariate Wasserstein distance between pred and target
    w2pred = torch.sqrt((pred_mean - target_mean) ** 2 + (pred_std - target_std) ** 2)  # Shape: (m,)

    # Compute the means and stds for ref (q reference distributions)
    ref_mean = torch.mean(ref, dim=1)  # Shape: (q,)
    ref_std = torch.std(ref, dim=1)  # Shape: (q,)

    # Broadcasting to compute Wasserstein distance between source[i] and ref[j] for all i, j
    # source_mean: (m,), ref_mean: (q,)
    w2ref = torch.sqrt((source_mean[:, None] - ref_mean[None, :]) ** 2 +
                       (source_std[:, None] - ref_std[None, :]) ** 2)  # Shape: (m, q)

    # Kernel smoothing for each reference distribution
    kernel = torch.exp(-w2ref ** 2 / (2 * bandwidth ** 2))  # Shape: (m, q)

    # Combine kernels by summing over all reference distributions
    combined_kernel = torch.sum(kernel, dim=1)  # Shape: (m,)

    # Smoothed loss: W2 distance between pred and target, weighted by the combined kernel
    smoothed_loss = w2pred * combined_kernel  # Shape: (m,)

    # Sum over all samples
    return smoothed_loss.sum()


# Training Loop with Early Stopping and Regularization
num_epochs = 50
train_losses = []
val_losses = []

optimizer = optim.SGD(model.parameters(), lr=0.001)  # Adjusted learning rate
patience = 1  # Early stopping patience
best_val_loss = float('inf')
epochs_no_improve = 0

# Select one sample from train and test dataset for visualization
train_sample_idx = 10
test_sample_idx = 10
train_source_sample = source_train[train_sample_idx]
train_target_sample = target_train[train_sample_idx]
test_source_sample = source_test[test_sample_idx]
test_target_sample = target_test[test_sample_idx]

# Store predictions for visualization
train_predictions_over_epochs = []
test_predictions_over_epochs = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for source_batch, target_batch in train_loader:
        optimizer.zero_grad()
        output = model(source_batch)
        loss = gaussian_w2(ref_data, source_batch, output, target_batch, bandwidth=1)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for source_batch, target_batch in test_loader:
            output = model(source_batch)
            loss = gaussian_w2(ref_data, source_batch, output, target_batch, bandwidth=1)
            val_loss += loss.item()
    val_loss /= len(test_loader)
    val_losses.append(val_loss)

    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Check early stopping condition
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print("Early stopping!")
        break

    # Store predictions for the selected train and test samples
    with torch.no_grad():
        train_prediction = model(torch.Tensor(train_source_sample).unsqueeze(0)).squeeze().numpy()
        test_prediction = model(torch.Tensor(test_source_sample).unsqueeze(0)).squeeze().numpy()
        train_predictions_over_epochs.append(train_prediction)
        test_predictions_over_epochs.append(test_prediction)

# Results Analysis
# Plotting Loss Curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
# Save the loss plot as a .jpg image with the current date attached
file_name = f'PWSM_loss_curves_{current_date}.jpg'
plt.savefig(file_name, format='jpg')
print(f"Saved loss curve as {file_name}")
plt.show()

# Create animation
def animate_distributions(target_sample, predictions_over_epochs, title):
    fig, ax = plt.subplots(figsize=(12, 6))

    def update(epoch):
        ax.clear()
        prediction = predictions_over_epochs[epoch].reshape(-1, 1)
        # source_sample_reshaped = source_sample.reshape(-1, 1)
        target_sample_reshaped = target_sample.reshape(-1, 1)
        # ax.hist(source_sample_reshaped, bins=30, alpha=0.5, label='Source', color='blue')
        ax.hist(target_sample_reshaped, bins=30, alpha=0.5, label='Target', color='red')
        ax.hist(prediction, bins=30, alpha=0.5, label=f'Predicted (Epoch {epoch + 1})')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{title} - Epoch {epoch + 1}')
        ax.legend()

    ani = FuncAnimation(fig, update, frames=len(train_predictions_over_epochs), repeat=False,
                        save_count=len(train_predictions_over_epochs))
    plt.show()
    return ani

def plot_and_save_distributions(target_sample, predictions_over_epochs, title, epochs, row_index, axs, save_prefix):
    """ Function to plot and save target vs predicted distributions for specified epochs. """

    # Ensure the directory exists before saving the images
    save_dir = os.path.dirname(save_prefix)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, epoch in enumerate(epochs):
        ax = axs[row_index, i]
        prediction = predictions_over_epochs[epoch - 1].reshape(-1, 1)  # Epochs are 1-indexed
        target_sample_reshaped = target_sample.reshape(-1, 1)

        ax.hist(target_sample_reshaped, bins=30, alpha=0.5, label='Target', color='red')
        ax.hist(prediction, bins=30, alpha=0.5, label=f'Predicted (Epoch {epoch})', color='blue')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Epoch {epoch}')
        ax.legend()

        # Save the figure for each subplot as a .jpg file
        file_name = f"{save_prefix}_epoch_{epoch}.jpg"
        plt.savefig(file_name, format='jpg')
        print(f"Saved {file_name}")

# Example usage (this part remains the same as before)
epochs_to_plot = [1, 2, len(train_predictions_over_epochs)]  # 1, 2, and final epoch

# Create the figure and axes (2 rows for train/test and 3 columns for the epochs)
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Train and Test Distributions at Epochs 1, 2, and Final', fontsize=16)

# Plot and save the train distributions (row 0 in the subplot grid)
plot_and_save_distributions(train_target_sample, train_predictions_over_epochs,
                            'Train Distributions', epochs_to_plot, row_index=0, axs=axs, save_prefix='train_images/train')

# Plot and save the test distributions (row 1 in the subplot grid)
plot_and_save_distributions(test_target_sample, test_predictions_over_epochs,
                            'Test Distributions', epochs_to_plot, row_index=1, axs=axs, save_prefix='test_images/test')

# Adjust layout to fit the subplots neatly
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()