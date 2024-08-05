'''
Limitations:
1. Complex Architecture

2. Performance
'''

def spatial_path(x):
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = MaxPooling2D((2, 2))(x)
    return x

def context_path(x):
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = AveragePooling2D(pool_size=(8, 8))(x)  # Global pooling
    x = Conv2D(256, (1, 1), padding='same', activation='relu')(x)
    x = UpSampling2D(size=(32, 32))(x)  # Adjust size based on input dimensions
    return x

def feature_fusion(sp, cp):
    x = concatenate([sp, cp], axis=-1)
    x = Conv2D(256, (1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    return x

def bisenet(input_shape=(256, 256, 3), num_classes=21):
    inputs = Input(input_shape)
    
    # Spatial Path
    sp_features = spatial_path(inputs)
    
    # Context Path
    cp_features = context_path(inputs)
    
    # Feature Fusion
    fused_features = feature_fusion(sp_features, cp_features)
    
    # Decoder
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(fused_features)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    # Final output
    x = Conv2D(num_classes, (1, 1), activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=x)
    
    return model
