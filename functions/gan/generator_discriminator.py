import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, concatenate
from tensorflow.keras.models import Model

def build_generator(latent_dim, num_classes):
    """
    Build the generator model for a GAN.

    Parameters:
    - latent_dim (int): The dimension of the input noise vector.
    - num_classes (int): The number of classes for conditional generation.

    Returns:
    - model (tf.keras.Model): The generator model.

    """
    inputs = Input(shape=(latent_dim,))
    labels = Input(shape=(num_classes,))

    x = concatenate([inputs, labels])

    x = Dense(64 * 64 * 32, activation='relu')(x)  # Adjusted for 512x512 input
    x = Reshape((64, 64, 32))(x)  # Adjusted for 512x512 input

    x = Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', activation='relu')(x)
    x = Conv2DTranspose(32, kernel_size=4, strides=2, padding='same', activation='relu')(x)

    outputs = Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='sigmoid')(x)  # Adjusted for 512x512 output

    model = Model([inputs, labels], outputs)
    return model

def build_discriminator(input_shape, num_classes):
    """
    Build the discriminator model.

    Args:
        input_shape (tuple): The shape of the input tensor (excluding batch dimension).
        num_classes (int): The number of classes.

    Returns:
        tf.keras.Model: The discriminator model.

    """
    inputs = Input(shape=input_shape)
    labels = Input(shape=(num_classes,))

    x = Conv2D(32, kernel_size=4, strides=2, padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(64, kernel_size=4, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Flatten()(x)
    x = concatenate([x, labels])

    x = Dense(1, activation='sigmoid')(x)

    model = Model([inputs, labels], x)
    return model
