import numpy as np

def normalize_tiles(tiles):
    """
    Normalize the given list of tiles by scaling their values between 0 and 1 using min-max normalization.
    
    Args:
        tiles (list of numpy.ndarray): The input list of images to be normalized.
        
    Returns:
        list of numpy.ndarray: The normalized images.
    """
    normalized_tiles = []
    
    for tile in tiles:
        # Convert to NumPy array if needed
        tile = np.asarray(tile, dtype=np.float64)

        # Min-max normalization to scale values between 0 and 1
        min_value = np.min(tile)
        max_value = np.max(tile)

        normalized_tile = (tile - min_value) / (max_value - min_value + 1e-20)  # Adding a small value to avoid division by zero

        normalized_tiles.append(normalized_tile)
    
    return normalized_tiles
