from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate

def unet_dual():
    """
    Creates a modified U-Net model accepting color images and binary masks with adjusted pooling to ensure compatible shapes.
    """
    # Input for color images
    image_input = Input((512, 512, 3))

    # Input for binary masks
    mask_input = Input((512, 512, 1))

    # --- Color Image Branch ---
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(image_input)
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # --- Mask Branch (Adjusted Pooling Size)---
    mask_conv1 = Conv2D(16, 3, activation='relu', padding='same')(mask_input)
    mask_conv1 = Conv2D(16, 3, activation='relu', padding='same')(mask_conv1)
    mask_pool1 = MaxPooling2D(pool_size=(2, 2))(mask_conv1)  # Maintains spatial resolution

    mask_conv2 = Conv2D(32, 3, activation='relu', padding='same')(mask_pool1)
    mask_conv2 = Conv2D(32, 3, activation='relu', padding='same')(mask_conv2)
    # No additional pooling here to avoid downsampling

    # --- Merging (shapes should now be compatible) ---
    merge1 = concatenate([conv2, mask_conv2], axis=3)
    
    # Decoder
    # Decoder
    up3 = UpSampling2D(size=(2, 2))(merge1) 
    conv4 = Conv2D(64, 3, activation='relu', padding='same')(up3)  
    conv4 = Conv2D(64, 3, activation='relu', padding='same')(conv4)  

    up5 = UpSampling2D(size=(2, 2))(conv2)  # Add upsampling here
    merge2 = concatenate([up5, conv4], axis=3)  # Skip connection
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(merge2) 

    # ... More decoder layers if needed...

    # Output layer
    outputs = Conv2D(1, 1, activation='sigmoid', padding='same')(conv5)  # Adjust filters for binary output

    model = Model(inputs=[image_input, mask_input], outputs=[outputs])
    return model
