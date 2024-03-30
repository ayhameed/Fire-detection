import cv2
import os
import numpy as np

def load_binary_masks(mask_dir):
    """
    Load binary mask images from a mask_dir into a NumPy array.

    Args:
        mask_dir (str): Path to the mask_dir containing mask images.

    Returns:
        numpy.ndarray: Array containing loaded binary masks.
    """
    # Get list of mask filenames sorted by their numeric part
    mask_filenames = sorted([filename for filename in os.listdir(mask_dir) if filename.endswith('.png')], key=lambda x: int(x.split('_')[1].split('.')[0]))

    # Initialize an empty list to store loaded masks
    masks = []

    # Iterate over mask filenames and load each mask
    for filename in mask_filenames:
        mask_path = os.path.join(mask_dir, filename)
        # Load mask as grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # Add an extra dimension to make it (512, 512, 1)
        mask = np.expand_dims(mask, axis=-1)
        # Append to the list of masks
        masks.append(mask)

    # Convert the list of masks into a NumPy array
    masks_array = np.array(masks)

    return masks_array
