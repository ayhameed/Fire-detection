import cv2
import time

# Function to Load image and visualize it

# import required libraries
import matplotlib.pyplot as plt

# Load the image with OpenCV
def load_image(imagePath):
    """
    Loads an image from a path, plots it, and returns a NumPy array.

    Args:
        imagePath (str): Path to the image file.

    Returns:
        np.ndarray: The loaded image as a NumPy array.
    """

    start_time = time.time()

    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    return image
