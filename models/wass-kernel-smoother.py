import torch
import torch.nn as nn
import torch.optim as optim
import geomloss

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

# TODO: Define Kh and compute it to feed into custom_loss
# TODO: wtf
# TODO: more.
# TODO: What is T_mu0 and mu_i here? How do they relate to T_mu0#mu_i?
# Custom loss function incorporating Wasserstein-2 distance and kernel smoother
def custom_loss(T_mu0, mu_i, nu_i, Kh):
    # computes the loss locally
    # computes W2 distance between T_mu0#mu_i and nu_i using sinkhorn divergence
    # blur param bw sinkhorn and kernel loss
    sink_loss = geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=0.0)
    wasserstein_distance = sink_loss(mu_i.view(1, -1, 1), nu_i.view(1, -1, 1))

    # Compute the kernel smoother term
    smoother_term = torch.sum(torch.pow(T_mu0(mu_i) - nu_i, 2) * Kh)

    # Combine the Wasserstein-2 distance and the kernel smoother term
    total_loss =  # Combine your Wasserstein-2 loss and smoother term appropriately

    return sink_loss

# TODO: what should these values be?
# Example usage in training loop
input_size =  # Set your input size
hidden_size =  # Set your hidden size
output_size =  # Set your output size

model = OTMapNN(input_size, hidden_size, output_size)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

num_epochs = 100

for epoch in range(num_epochs):
    # Assuming inputs, mu_i, nu_i, and Kh are your data and kernel smoother parameters
    inputs = torch.tensor(input_data, requires_grad=True)
    mu_i = torch.tensor(mu_i_data, requires_grad=False)
    nu_i = torch.tensor(nu_i_data, requires_grad=False)
    Kh = torch.tensor(Kh_data, requires_grad=False)

    # Forward pass
    T_mu0 = model(inputs)
    
    # Compute loss
    loss = custom_loss(T_mu0, mu_i, nu_i, Kh)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')
