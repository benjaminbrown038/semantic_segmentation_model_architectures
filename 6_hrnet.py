'''
Limitations:
1. High Computational Cost

2. Memory Requirements
''' 

def knowledge_representation_module(x):
    # Apply convolution layers to extract contextual information
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    return x

def encoder(inputs):
    x = Conv2D(64, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    pool1 = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(128, (3, 3), padding='same')(pool1)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    pool2 = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(256, (3, 3), padding='same')(pool2)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    pool3 = MaxPooling2D((2, 2))(x)
    
    return pool3

def decoder(encoded, knowledge_representation):
    x = Conv2D(256, (3, 3), padding='same')(encoded)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = UpSampling2D((2, 2))(x)
    x = concatenate([x, knowledge_representation])
    
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = UpSampling2D((2, 2))(x)
    
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    return x

def krnet(input_shape=(256, 256, 3), num_classes=21):
    inputs = Input(input_shape)
    
    # Encoder
    encoded = encoder(inputs)
    
    # Knowledge Representation
    knowledge = knowledge_representation_module(encoded)
    
    # Decoder
    decoded = decoder(encoded, knowledge)
    
    # Output layer
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(decoded)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model
