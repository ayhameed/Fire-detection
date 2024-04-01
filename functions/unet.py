from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout

def unet():
    """
    Creates a U-Net model with dropout regularization for image segmentation.

    Returns:
        A Keras Model object representing the U-Net model with dropout.
    """
    inputs = Input((512, 512, 3))

    # Encoder
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(0.001)(pool1)

    conv2 = Conv2D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(0.001)(pool2)

    # Bottleneck
    conv3 = Conv2D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, 3, activation='relu', padding='same')(conv3)

    # Decoder
    up3 = UpSampling2D(size=(2, 2))(conv3)
    conv4 = Conv2D(64, 3, activation='relu', padding='same')(up3)
    conv4 = Conv2D(64, 3, activation='relu', padding='same')(conv4)
    conv4 = Dropout(0.001)(conv4)

    merge1 = concatenate([conv2, conv4], axis=3)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(merge1)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(conv5)
    conv5 = Dropout(0.001)(conv5)

    up2 = UpSampling2D(size=(2, 2))(conv5)
    conv6 = Conv2D(32, 3, activation='relu', padding='same')(up2)
    conv6 = Conv2D(32, 3, activation='relu', padding='same')(conv6)
    conv6 = Dropout(0.001)(conv6)

    merge2 = concatenate([conv1, conv6], axis=3)
    conv7 = Conv2D(32, 3, activation='relu', padding='same')(merge2)
    conv7 = Conv2D(32, 3, activation='relu', padding='same')(conv7)

    # Output layer
    outputs = Conv2D(1, 1, activation='sigmoid', padding='same')(conv7)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model
