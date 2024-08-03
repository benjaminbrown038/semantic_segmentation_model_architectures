
def fcn(input_shape=(256, 256, 3), num_classes=21):
    inputs = Input(input_shape)

    # Encoder (Feature extraction)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    p1 = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), padding='same', activation='relu')(p1)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    p2 = MaxPooling2D((2, 2))(x)

    x = Conv2D(256, (3, 3), padding='same', activation='relu')(p2)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    p3 = MaxPooling2D((2, 2))(x)

    x = Conv2D(512, (3, 3), padding='same', activation='relu')(p3)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    p4 = MaxPooling2D((2, 2))(x)

    # Classification Layer
    x = Conv2D(4096, (7, 7), padding='same', activation='relu')(p4)
    x = Conv2D(4096, (1, 1), padding='same', activation='relu')(x)
    x = Conv2D(num_classes, (1, 1), padding='same')(x)
    
    # Upsampling
    x = UpSampling2D((32, 32))(x)  # Upsample to original image size
    
    # Final Output
    outputs = x

    model = Model(inputs=inputs, outputs=outputs)

    return model
