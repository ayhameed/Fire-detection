import numpy as np

def tile_image(normalized_image, tile_size, overlap):
    """
    Creates image tiles with optional overlap, padding incomplete tiles with zeros.

    Args:
        normalized_image (numpy.ndarray): The normalized image as a NumPy array.
        tile_size (tuple): A tuple (height, width) specifying the desired tile size.
        overlap (tuple): A tuple (vertical_overlap, horizontal_overlap) specifying the overlap between tiles.

    Returns:
        list: A list of tiled images as NumPy arrays.
    """

    image = normalized_image.copy()  # Make a copy to avoid modifying the original

    height, width = image.shape[:2]
    tile_height, tile_width = tile_size
    vert_overlap, horiz_overlap = overlap

    # Calculate the number of tiles needed, accounting for potential edge cases
    num_vert_tiles = int(np.ceil((height - tile_height) / (tile_height - vert_overlap)) + 1)
    num_horiz_tiles = int(np.ceil((width - tile_width) / (tile_width - horiz_overlap)) + 1)

    tiles = []
    for i in range(num_vert_tiles):
        start_y = i * (tile_height - vert_overlap)
        end_y = min(start_y + tile_height, height)

        for j in range(num_horiz_tiles):
            start_x = j * (tile_width - horiz_overlap)
            end_x = min(start_x + tile_width, width)

            tile = image[start_y:end_y, start_x:end_x]

            # Pad if necessary
            if tile.shape[0] < tile_height:
                pad_y = tile_height - tile.shape[0]
                tile = np.pad(tile, ((0, pad_y), (0, 0), (0, 0)), mode='constant')
            if tile.shape[1] < tile_width:
                pad_x = tile_width - tile.shape[1]
                tile = np.pad(tile, ((0, 0), (0, pad_x), (0, 0)), mode='constant')

            tiles.append(tile)

    return tiles
