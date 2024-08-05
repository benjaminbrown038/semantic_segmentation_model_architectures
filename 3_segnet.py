'''
Limitations:

1. Slow Inference

2. Complex Architecture
'''

def segnet(input_shape=(256, 256, 3), num_classes=21):
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1, pool1_mask = MaxPooling2D((2, 2), return_indices=True)(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2, pool2_mask = MaxPooling2D((2, 2), return_indices=True)(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3, pool3_mask = MaxPooling2D((2, 2), return_indices=True)(conv3)

    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    pool4, pool4_mask = MaxPooling2D((2, 2), return_indices=True)(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    # Decoder
    unpool5 = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(conv5)
    unpool5 = concatenate([unpool5, conv4])
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(unpool5)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    unpool4 = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(conv6)
    unpool4 = concatenate([unpool4, conv3])
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(unpool4)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    unpool3 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(conv7)
    unpool3 = concatenate([unpool3, conv2])
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(unpool3)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    unpool2 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(conv8)
    unpool2 = concatenate([unpool2, conv1])
    conv9 = Conv
