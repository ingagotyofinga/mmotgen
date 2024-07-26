import numpy as np
import matplotlib.pyplot as plt
import ot


# Define the source and target distributions
P = np.array([[0, 0.4], [1, 0.6]])
Q = np.array([[2, 0.5], [3, 0.5]])

# Points and masses
x = P[:, 0]
p = P[:, 1]
y = Q[:, 0]
q = Q[:, 1]

# Cost matrix (Euclidean distance)
C = ot.dist(x.reshape(-1, 1), y.reshape(-1, 1))

# Compute the optimal transport plan
T = ot.emd(p, q, C)

# Print the optimal transport plan
print("Optimal transport plan:\n", T)

# Visualize the distributions and the optimal transport plan
plt.figure(figsize=(10, 6))

# Plot source distribution
for xi, pi in zip(x, p):
    plt.scatter(xi, 0, s=1000 * pi, c='blue', alpha=0.5, label='Source' if xi == 0 else "")

# Plot target distribution
for yi, qi in zip(y, q):
    plt.scatter(yi, 1, s=1000 * qi, c='red', alpha=0.5, label='Target' if yi == 2 else "")

# Plot the transport plan
for i in range(len(x)):
    for j in range(len(y)):
        if T[i, j] > 0:
            plt.plot([x[i], y[j]], [0, 1], 'k-', alpha=T[i, j])

# Add labels and legend
plt.xlabel('Position')
plt.ylabel('Distribution')
plt.title('Optimal Transport Plan')
plt.legend()
plt.grid()

# Show the plot
plt.show()
