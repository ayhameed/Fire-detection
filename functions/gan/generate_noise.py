import numpy as np


# Generate random noise data
def generate_noise_data(num_samples, dim):
    return np.random.normal(0, 1, size=(num_samples, dim))

