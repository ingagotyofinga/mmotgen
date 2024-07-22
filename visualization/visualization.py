import numpy as np
import matplotlib.pyplot as plt
from data.simulate_data_old import DataSimulatorOld
from mpl_toolkits.mplot3d import Axes3D

class DataVisualizer:
    def __init__(self, num_distributions, num_bins, num_dimensions):
        self.num_distributions = num_distributions
        self.num_bins = num_bins
        self.num_dimensions = num_dimensions

    def visualize(self, data, title="Data Visualization", labels=None, colors=None):
        if self.num_dimensions == 1:
            self._visualize_1d(data, title, labels, colors)
        elif self.num_dimensions == 2:
            self._visualize_2d(data, title, labels, colors)
        elif self.num_dimensions == 3:
            self._visualize_3d(data, title, labels, colors)
        else:
            raise ValueError("Unsupported number of dimensions: {}".format(self.num_dimensions))

    def _visualize_1d(self, data, title, labels, colors):
        plt.figure(figsize=(10, 6))
        for idx, dataset in enumerate(data):
            color = colors[idx] if colors else None
            for i in range(dataset.shape[0]):
                plt.plot(dataset[i], label=f'{labels[idx]} Distribution {i+1}' if labels else f'Distribution {i+1}', color=color)
        plt.title(title)
        plt.xlabel('Bins')
        plt.ylabel('Values')
        plt.legend()
        plt.show()

    def _visualize_2d(self, data, title, labels, colors):
        plt.figure(figsize=(10, 6))
        for idx, dataset in enumerate(data):
            color = colors[idx] if colors else None
            for i in range(dataset.shape[0]):
                plt.scatter(dataset[i, :, 0], dataset[i, :, 1], label=f'{labels[idx]} Distribution {i+1}' if labels else f'Distribution {i+1}', color=color)
        plt.title(title)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        # plt.legend()
        plt.show()

    def _visualize_3d(self, data, title, labels, colors):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        for idx, dataset in enumerate(data):
            color = colors[idx] if colors else None
            for i in range(dataset.shape[0]):
                ax.scatter(dataset[i, :, 0], dataset[i, :, 1], dataset[i, :, 2], label=f'{labels[idx]} Distribution {i+1}' if labels else f'Distribution {i+1}', color=color)
        ax.set_title(title)
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
        # ax.legend()
        plt.show()

# Example usage:
if __name__ == "__main__":
    # Simulate some data for demonstration purposes
    num_distributions = 10
    num_bins = 10
    num_dimensions = 3

    means = np.random.randint(low=1, high=500, size=(num_distributions, num_dimensions))
    std_devs = np.random.randint(low=1, high=50, size=(num_distributions, num_dimensions))

    # Generate two sets of random data
    simulator = DataSimulator(num_distributions, num_bins, num_dimensions)
    data1 = simulator.generate_source_data(means, std_devs)
    data2, step = simulator.generate_mu0_distributions(data1)

    # Combine data into a list and define labels and colors
    data = [data1, data2]
    labels = ["Dataset 1", "Dataset 2"]
    colors = ['blue', 'red']  # Color for each dataset

    # Create an instance of DataVisualizer
    visualizer = DataVisualizer(num_distributions, num_bins, num_dimensions)

    # Visualize the data
    visualizer.visualize(data, title="Randomly Generated Data", labels=labels, colors=colors)
