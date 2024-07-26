





import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.model_selection import train_test_split

# Seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Data Generation
def generate_gaussian_data(num_samples, mean, std, num_distributions):
    data = []
    for _ in range(num_distributions):
        samples = np.random.normal(loc=mean, scale=std, size=num_samples)
        data.append(samples)
    return np.array(data)

num_samples = 1000
num_distributions = 10000
source_mean = 0
source_std = 1
target_mean = 7  # Shifted to 7 to be significantly different from source
target_std = 2

source_data = generate_gaussian_data(num_samples, source_mean, source_std, num_distributions)
target_data = generate_gaussian_data(num_samples, target_mean, target_std, num_distributions)

source_data = (source_data - np.mean(source_data, axis=0)) / np.std(source_data, axis=0)
target_data = (target_data - np.mean(target_data, axis=0)) / np.std(target_data, axis=0)

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
class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
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

# Gaussian Wasserstein Loss Function with Kernel Smoothing
def gaussian_wasserstein_loss(pred, target, bandwidth=4):
    pred_mean = torch.mean(pred, dim=1)
    pred_std = torch.std(pred, dim=1)
    target_mean = torch.mean(target, dim=1)
    target_std = torch.std(target, dim=1)

    wasserstein_distance = torch.sqrt((pred_mean - target_mean) ** 2 + (pred_std - target_std) ** 2)

    # Kernel smoothing
    kernel = torch.exp(-wasserstein_distance ** 2 / (2 * bandwidth ** 2))
    smoothed_loss = wasserstein_distance * kernel

    return smoothed_loss.sum()

# Training Loop with Early Stopping and Regularization
num_epochs = 50
train_losses = []
val_losses = []

optimizer = optim.SGD(model.parameters(), lr=0.01)  # Adjusted learning rate
patience = 10  # Early stopping patience
best_val_loss = float('inf')
epochs_no_improve = 0

# Select one sample from train and test dataset for visualization
train_sample_idx = 0
test_sample_idx = 0
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
        loss = gaussian_wasserstein_loss(output, target_batch)
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
            loss = gaussian_wasserstein_loss(output, target_batch)
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

    ani = FuncAnimation(fig, update, frames=num_epochs, repeat=False)
    plt.show()
    return ani

# Generate animations
train_animation = animate_distributions(train_target_sample, train_predictions_over_epochs,
                                        'Evolution of Train Distributions')
test_animation = animate_distributions(test_target_sample, test_predictions_over_epochs,
                                       'Evolution of Test Distributions')

# Save animations
train_animation.save('train_animation.gif', writer='imagemagick')
test_animation.save('test_animation.gif', writer='imagemagick')
