import cv2
import numpy as np

def rotate_tiles(tiles, angle):
    """
    Rotate a list of input tiles by the specified angle.

    Args:
        tiles (list): List of input tiles as NumPy arrays.
        angle (float): The angle of rotation in degrees.

    Returns:
        list: List of rotated tiles.
    """
    rotated_tiles = []
    for tile in tiles:
        # Get image center coordinates
        center = tuple(np.array(tile.shape[1::-1]) / 2)
        # Perform rotation
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_tile = cv2.warpAffine(tile, rotation_matrix, tile.shape[1::-1], flags=cv2.INTER_LINEAR)
        rotated_tiles.append(rotated_tile)
    return rotated_tiles
