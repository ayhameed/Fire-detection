import cv2
import numpy as np

def rgb_to_binary(rgb_masks, threshold=128):
    """
    Converts RGB masks to binary masks using a specified threshold.

    Args:
        rgb_masks (list): List of RGB masks.
        threshold (int): Threshold value for binary conversion. Default is 128.

    Returns:
        list: List of binary masks.
    """
    binary_masks = []
    for rgb_mask in rgb_masks:
        # Convert to NumPy array if needed
        mask_array = np.asarray(rgb_mask, dtype=np.float64)

        # Min-max normalization to scale values between 0 and 1
        min_value = np.min(mask_array)
        max_value = np.max(mask_array)

        normalized_mask = (mask_array - min_value) / (max_value - min_value + 1e-20)  # Adding a small value to avoid division by zero

        # Extract red channel
        red_channel = normalized_mask[:, :, 0]
        
        # Apply thresholding
        _, binary_mask = cv2.threshold(red_channel, threshold, 1, cv2.THRESH_BINARY)
        
        binary_masks.append(binary_mask)
    
    return binary_masks
