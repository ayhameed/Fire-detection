import os
import cv2
import numpy as np

def load_images_from_directory(directory, prefix="tile_"):
    """
    Load RGB images from a directory based on their names and return them as a list of NumPy arrays.

    Args:
        directory (str): The directory containing the image files.
        prefix (str): The prefix used in the image filenames (default: "tile_").

    Returns:
        list: A list of RGB images, each as a NumPy array.
    """
    # Check if the directory exists
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory '{directory}' not found.")

    # Initialize an empty list to store the images
    images = []

    # Get list of image files sorted by name
    image_files = sorted([file for file in os.listdir(directory) if file.startswith(prefix) and any(file.endswith(ext) for ext in ['.jpg', '.jpeg', '.png'])],
                         key=lambda x: int(x[len(prefix):-len('.png')]))

    # Iterate over all files in the directory
    for filename in image_files:
        file_path = os.path.join(directory, filename)

        # Load the image using OpenCV
        image = cv2.imread(file_path)

        # Check if the image was successfully loaded
        if image is None:
            print(f"Warning: Unable to load image '{filename}'. Skipping...")
            continue

        # Check if the loaded image has three channels (RGB)
        if image.shape[2] == 3:
            images.append(image)
        else:
            print(f"Warning: Image '{filename}' is not RGB. Skipping...")

    return images
