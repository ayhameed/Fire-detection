import numpy as np
import matplotlib.pyplot as plt

def img_threshold(normalized_image_copy, threshold):
    """
    Applies thresholding to each RGB band of an image and plots the original image and thresholded bands.

    Args:
        normalized_image_copy (numpy.ndarray): Normalized image array.
        threshold (float): Threshold value for thresholding.

    Returns:
        None
    """
    # Splitting the image into RGB bands
    image = normalized_image_copy
    red_band = image[:, :, 0]
    green_band = image[:, :, 1]
    blue_band = image[:, :, 2]

    # Applying threshold to each band
    red_thresholded = red_band > threshold
    green_thresholded = green_band > threshold
    blue_thresholded = blue_band > threshold

    # Plotting
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))
    axs[0].imshow(image)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(red_thresholded, cmap='hot')  # Using 'hot' colormap for better visibility
    axs[1].set_title('Red Thresholded')
    axs[1].axis('off')

    axs[2].imshow(green_thresholded, cmap='hot')  # Using 'hot' colormap for better visibility
    axs[2].set_title('Green Thresholded')
    axs[2].axis('off')

    axs[3].imshow(blue_thresholded, cmap='hot')  # Using 'hot' colormap for better visibility
    axs[3].set_title('Blue Thresholded')
    axs[3].axis('off')

    plt.tight_layout()
    plt.show()
