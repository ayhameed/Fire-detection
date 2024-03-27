import cv2
import numpy as np
import os

def create_masks(tile_dir="../dataset/tiles", mask_dir="../dataset/masks", threshold=0.3):
    """
    Loads image tiles, extracts the red band, applies a threshold, and saves the masks.

    Args:
        tile_dir (str): The directory containing the image tiles.
        mask_dir (str): The directory to save the generated masks.
        threshold (float): The threshold value to apply to the red band (default: 0.3).

    Returns:
        list: A list of the created mask images as NumPy arrays.
    """

    # Create mask directory if it doesn't exist
    os.makedirs(mask_dir, exist_ok=True)

    masks = []
    for filename in os.listdir(tile_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):  # Check for image files 
            filepath = os.path.join(tile_dir, filename)
            image = cv2.imread(filepath)

            # Extract the red band
            red_band = image[:, :, 0]  

            # Apply threshold
            red_thresholded = red_band > threshold

            # Save the mask
            mask_filename = os.path.splitext(filename)[0] + "_mask.png"  # Generate mask name
            mask_path = os.path.join(mask_dir, mask_filename)
            cv2.imwrite(mask_path, red_thresholded.astype(np.uint8) * 255)  # Save as a binary image

            masks.append(red_thresholded)  # Store the mask

    return masks 