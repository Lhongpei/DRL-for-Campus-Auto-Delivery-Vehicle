import numpy as np

def select_k_coordinates(grid_size, k, mask=None):
    # Create an array of all possible indices in the grid
    indices = np.arange(grid_size[0] * grid_size[1])

    if mask is not None:
        # Convert mask coordinates to linear indices
        mask_indices = np.array([x * grid_size[1] + y for x, y in mask])
        # Remove the mask indices from the possible indices
        indices = np.setdiff1d(indices, mask_indices)
    
    # Randomly select k indices from the remaining ones
    selected_indices = np.random.choice(indices, size=k, replace=False)

    # Convert linear indices back to coordinates
    coordinates = np.unravel_index(selected_indices, grid_size)
    coordinates = list(zip(coordinates[0], coordinates[1]))
    
    return coordinates

def normal_weight(grid_size, mean=0.5, std=0.1):
    return np.random.normal(mean, std, grid_size)

def uniform_weight(grid_size, low=0, high=1):
    return np.random.uniform(low, high, grid_size)

class NormalWeightGrid:
    def __init__(self, grid_size, mean_range=(0.4, 0.6), std_range=(0.05, 0.15)):
        self.grid_size = grid_size
        self.mean = np.random.uniform(mean_range[0], mean_range[1], grid_size)
        self.std = np.random.uniform(std_range[0], std_range[1], grid_size)

    def generate_weights(self):
        return np.random.normal(self.mean, self.std)

    def get_mean_std(self):
        return self.mean, self.std