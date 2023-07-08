# A script for loading and preparing the dataset for training or evaluation.
# data/data_loader.py
import numpy as np


class DataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(dataset)
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        self.reset()

    def reset(self):
        self.indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current_batch = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_batch >= self.num_batches:
            self.reset()
            raise StopIteration
        else:
            start_idx = self.current_batch * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            batch_indices = self.indices[start_idx:end_idx]
            batch_data = [self.dataset[idx] for idx in batch_indices]
            self.current_batch += 1
            return batch_data

# def load_data():
# Load your data here, e.g., from files or using existing libraries
# Preprocess the data as needed (e.g., normalization, resizing, etc.)
# Split the data into train, validation, and test sets if necessary
# Return the data splits or any other required data structures

# Example implementation
# train_data = ...
# val_data = ...
# test_data = ...

# return train_data, val_data, test_data
