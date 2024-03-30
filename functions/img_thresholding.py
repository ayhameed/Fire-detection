import numpy as np
import matplotlib.pyplot as plt

def img_threshold(normalized_image_copy, threshold):
    """
    Applies the same threshold value to each RGB band of an image and plots the binary thresholded images for each band.

    Args:
        normalized_image_copy (numpy.ndarray): Normalized image array.
        threshold (float): Threshold value for all RGB bands.

    Returns:
        None
    """
    # Applying the same threshold value across all RGB bands
    image = normalized_image_copy
    red_band = image[:, :, 0]
    green_band = image[:, :, 1]
    blue_band = image[:, :, 2]

    # Thresholding each RGB band separately
    red_thresholded = red_band > threshold
    green_thresholded = green_band > threshold
    blue_thresholded = blue_band > threshold

    # Plotting
    fig, axs = plt.subplots(1, 4, figsize=(16, 5))
    
    # Original Image
    axs[0].imshow(normalized_image_copy)
    axs[0].set_title('Original')
    axs[0].axis('off')
    
    # Red Band
    axs[1].imshow(red_thresholded, cmap='binary')
    axs[1].set_title('Red Mask')
    axs[1].axis('off')

    # Green Band
    axs[2].imshow(green_thresholded, cmap='binary')
    axs[2].set_title('Green Mask')
    axs[2].axis('off')

    # Blue Band
    axs[3].imshow(blue_thresholded, cmap='binary')
    axs[3].set_title('Blue Mask')
    axs[3].axis('off')

    plt.tight_layout()
    plt.show()
