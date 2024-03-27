#import required libraries
import numpy as np
 
 
#function to normalize imagery
def normalize_image(image):
    """
    Normalize the given image by scaling its values between 0 and 1 using min-max normalization.
    
    Args:
        image (numpy.ndarray): The input image to be normalized.
        
    Returns:
        numpy.ndarray: The normalized image.
    """
    # Convert to NumPy array if needed
    image = np.asarray(image, dtype=np.float64)

    # Min-max normalization to scale values between 0 and 1
    min_value = np.min(image)
    max_value = np.max(image)

    normalized_image = (image - min_value) / (max_value - min_value + 1e-20)  # Adding a small value to avoid division by zero

    return normalized_image