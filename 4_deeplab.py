
def atrous_conv(x, filters, kernel_size, dilation_rate):
    x = Conv2D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def xception_backbone(input_shape):
    base_model = Xception(include_top=False, weights='imagenet', input_shape=input_shape)
    
    # Extract layers from Xception
    base_model.trainable = False
    outputs = [base_model.get_layer(name).output for name in ['block13_pool', 'block13_sepconv2_bn', 'block14_sepconv2_bn']]
    
    return base_model.input, outputs

def aspp(x):
    size = x.shape[1]
    b0 = atrous_conv(x, 256, (1, 1), dilation_rate=1)
    b1 = atrous_conv(x, 256, (3, 3), dilation_rate=6)
    b2 = atrous_conv(x, 256, (3, 3), dilation_rate=12)
    b3 = atrous_conv(x, 256, (3, 3), dilation_rate=18)
    
    # Global Average Pooling
    b4 = tf.keras.layers.GlobalAveragePooling2D()(x)
    b4 = tf.keras.layers.Reshape((1, 1, x.shape[-1]))(b4)
    b4 = atrous_conv(b4, 256, (1, 1), dilation_rate=1)
    b4 = UpSampling2D(size=(size, size))(b4)
    
    # Concatenate and apply a final convolution
    x = concatenate([b0, b1, b2, b3, b4], axis=-1)
    x = atrous_conv(x, 256, (1, 1), dilation_rate=1)
    
    return x

def deeplabv3(input_shape=(256, 256, 3), num_classes=21):
    inputs, [low_level_features, x, _] = xception_backbone(input_shape)
    
    # ASPP
    x = aspp(x)
    
    # Decoder
    x = UpSampling2D(size=(4, 4))(x)
    x = concatenate([x, low_level_features])
    x = atrous_conv(x, 256, (3, 3), dilation_rate=1)
    x = atrous_conv(x, 256, (3, 3), dilation_rate=1)
    
    # Final layer
    x = Conv2D(num_classes, (1, 1), activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=x)
    
    return model
