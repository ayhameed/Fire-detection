import cv2
import numpy as np

import matplotlib.pyplot as plt

def detect_fire(normalized_image, block_size, c):
    """
    Detects fire in an image using adaptive thresholding.

    Args:
        image_path (str): Path to the image file.
        block_size (int): Size of the neighborhood area for adaptive thresholding.
        c (int): Constant subtracted from the mean or weighted mean.

    Returns:
        None
    """
    # Read the image
    image = normalized_image.copy()
    
    # Split the image into individual RGB bands
    b, g, r = cv2.split(image)
    
    # Apply adaptive thresholding to each band to detect fire
    b_thresh = cv2.adaptiveThreshold(b, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c)
    g_thresh = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c)
    r_thresh = cv2.adaptiveThreshold(r, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c)
    
    # Plot the original image alongside the thresholded images
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))
    
    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original Image')
    
    axs[1].imshow(b_thresh, cmap='gray')
    axs[1].set_title('Thresholded Blue Band')
    
    axs[2].imshow(g_thresh, cmap='gray')
    axs[2].set_title('Thresholded Green Band')
    
    axs[3].imshow(r_thresh, cmap='gray')
    axs[3].set_title('Thresholded Red Band')
    
    plt.tight_layout()
    plt.show()
