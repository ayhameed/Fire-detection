import numpy as np
import tensorflow as tf  # or 'import keras' if using standalone Keras
from tensorflow.keras import backend as K  # Import Keras backend functions

def dice_loss(y_true, y_pred):
    """
    Dice loss function for image segmentation tasks.

    Args:
        y_true: Ground truth labels (one-hot encoded).
        y_pred: Model predictions (one-hot encoded).

    Returns:
        Dice loss (scalar).
    """
    smooth = 1.0  # Smoothing factor to avoid division by zero

    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    dice = K.mean((2.0 * intersection + smooth) / (union + smooth), axis=-1)
    return 1 - dice  # Return actual Dice loss (for minimization)