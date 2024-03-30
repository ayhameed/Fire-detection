import cv2
import numpy as np
import os

def create_masks(tiles, mask_dir="../dataset/masks", threshold=0.3):
    """
    Extracts the red band from image tiles, applies a threshold, and saves the masks.

    Args:
        tiles (list): List of image tiles as NumPy arrays.
        mask_dir (str): The directory to save the generated masks.
        threshold (float): The threshold value to apply to the red band (default: 0.3).

    Returns:
        list: A list of the created mask images as NumPy arrays.
    """

    # Create mask directory if it doesn't exist
    os.makedirs(mask_dir, exist_ok=True)

    masks = []
    for i, tile in enumerate(tiles):
        # Extract the red band
        red_band = tile[:, :, 0]  

        # Apply threshold
        red_thresholded = red_band > threshold

        # Add extra dimension to make it (512, 512, 1)
        red_thresholded = np.expand_dims(red_thresholded, axis=-1)

        # Save the mask
        mask_filename = f"mask_{i}.png"  # Generate mask name
        mask_path = os.path.join(mask_dir, mask_filename)
        
        # Save tile with debugging info 
        print(f"Saving to: {mask_path}") 
        cv2.imwrite(mask_path, red_thresholded.astype(np.uint8) * 255)  # Save as a binary image

        masks.append(red_thresholded)  # Store the mask

    return masks
