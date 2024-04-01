import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

def weighted_dice_loss(y_true, y_pred, weight=0.8):
    """
    Weighted Dice loss function (adjusted for single-channel masks).
    """
    smooth = 1.0

    # Calculate intersection (weighted by 'weight')
    intersection = K.sum(K.abs(y_true * y_pred) * weight, axis=-1)

    # Calculate union (weighted by 1 + 'weight')
    union = K.sum(K.abs(y_true) + K.abs(y_pred) * (1 + weight), axis=-1)

    # Calculate weighted Dice coefficient
    dice = K.mean((2.0 * intersection + smooth) / (union + smooth), axis=-1)

    return 1 - dice
