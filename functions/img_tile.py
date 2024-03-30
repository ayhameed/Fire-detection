import numpy as np
import cv2
import os

def tile_image(normalized_image, tile_size, overlap, save_directory="../dataset/tiles", 
               padding_value=0, output_format='png'):
    """
    Creates image tiles with optional overlap, padding incomplete tiles, and saves 
    them in the specified format.

    Args:
        normalized_image (numpy.ndarray): The normalized image as a NumPy array.
        tile_size (tuple): A tuple (height, width) specifying the desired tile size.
        overlap (tuple): A tuple (vertical_overlap, horizontal_overlap) specifying the overlap.
        save_directory (str): Directory to save the tiles.
        padding_value (int or float): Value to use for padding incomplete tiles (default: 0).
        output_format (str): Image format for saving tiles (e.g., 'png', 'jpg').
    """

    image = normalized_image.copy()  
    height, width = image.shape[:2]
    tile_height, tile_width = tile_size
    vert_overlap, horiz_overlap = overlap

    # Calculate the number of tiles (with potential edge cases)
    num_vert_tiles = int(np.ceil((height - tile_height) / (tile_height - vert_overlap)) + 1)
    num_horiz_tiles = int(np.ceil((width - tile_width) / (tile_width - horiz_overlap)) + 1)

    # Create save directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    tiles = []
    for i in range(num_vert_tiles):
        start_y = i * (tile_height - vert_overlap)
        end_y = min(start_y + tile_height, height)

        for j in range(num_horiz_tiles):
            start_x = j * (tile_width - horiz_overlap)
            end_x = min(start_x + tile_width, width)

            tile = image[start_y:end_y, start_x:end_x]

            # Pad if necessary (combined conditions for minor efficiency)
            if tile.shape[0] < tile_height or tile.shape[1] < tile_width:
                pad_y = tile_height - tile.shape[0]
                pad_x = tile_width - tile.shape[1]
                tile = np.pad(tile, ((0, pad_y), (0, pad_x), (0, 0)), mode='constant', constant_values=padding_value)

            # Save tile
            tile_filename = f"tile_{i * num_horiz_tiles + j}.{output_format}" 
            tile_path = os.path.join(save_directory, tile_filename)
            
            # Save tile with debugging info 
            print(f"Saving to: {tile_path}") 

            success = cv2.imwrite(tile_path, cv2.cvtColor((tile * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            if not success:
                print(f"Error saving tile: {tile_path}")

            tiles.append(tile)

    return tiles
