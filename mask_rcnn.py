def build_backbone(input_shape=(256, 256, 3)):
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    return base_model

def build_rpn(base_model):
    x = base_model.output

    # Shared convolutional layer
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    
    # RPN classification and regression
    rpn_class = Conv2D(9, (1, 1), activation='sigmoid')(x)
    rpn_regress = Conv2D(36, (1, 1))(x)

    return rpn_class, rpn_regress
  
def roi_align(feature_map, boxes, pool_size):
    # This is a simplified version of RoIAlign
    boxes = tf.cast(boxes, tf.float32)
    boxes = tf.image.crop_and_resize(feature_map, boxes, box_indices=tf.zeros(tf.shape(boxes)[0], dtype=tf.int32), crop_size=[pool_size, pool_size])
    return boxes

def build_mask_network(feature_map):
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(feature_map)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    mask_output = Conv2D(1, (1, 1), activation='sigmoid')(x)  # 1 for binary masks; change to num_classes for multi-class
    return mask_output


def build_mask_rcnn(input_shape=(256, 256, 3), num_classes=21):
    inputs = Input(input_shape)
    
    # Backbone network
    base_model = build_backbone(input_shape)
    
    # Feature extraction
    feature_map = base_model(inputs)
    
    # RPN
    rpn_class, rpn_regress = build_rpn(base_model)
    
    # RoIAlign (simplified version)
    rois = roi_align(feature_map, boxes, pool_size=14)  # Adjust boxes as needed

    # Mask Network
    mask_output = build_mask_network(rois)
    
    # Final mask prediction
    final_output = Conv2D(num_classes, (1, 1), activation='softmax')(mask_output)

    model = Model(inputs=inputs, outputs=[rpn_class, rpn_regress, final_output])
    
    return model

