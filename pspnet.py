def pyramid_pooling_module(x, bin_sizes):
    # Feature maps list to collect the output of each pyramid level
    features = []
    h, w = x.shape[1], x.shape[2]  # height and width of the input feature map

    # Original feature map
    features.append(x)

    for bin_size in bin_sizes:
        pool_size = (h // bin_size, w // bin_size)
        pooled = tf.keras.layers.AvgPooling2D(pool_size, strides=pool_size, padding='same')(x)
        # Resize to original size
        upsampled = tf.keras.layers.Conv2D(x.shape[-1], (1, 1), padding='same')(pooled)
        upsampled = tf.image.resize(upsampled, (h, w))
        features.append(upsampled)

    # Concatenate all pyramid features
    psp_feature = concatenate(features, axis=-1)
    return psp_feature



def pspnet(input_shape=(256, 256, 3), num_classes=21):
    inputs = Input(input_shape)

    # Backbone network (e.g., ResNet)
    # Here, using a simple CNN as a placeholder for the backbone
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)

    # Pyramid pooling module
    psp_features = pyramid_pooling_module(x, bin_sizes=[1, 2, 3, 6])

    # Final convolutional layer
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(psp_features)
    x = Conv2D(num_classes, (1, 1), activation='softmax')(x)  # For multi-class segmentation

    model = Model(inputs=[inputs], outputs=[x])

    return model
