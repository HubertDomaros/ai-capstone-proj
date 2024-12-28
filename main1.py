from keras.api.layers import (
    Input, Conv2D, Conv2DTranspose, MaxPooling2D, 
    SpatialDropout2D, concatenate, BatchNormalization, Activation
)
from keras.api.models import Model
from keras.api.utils import Sequence

import xml.etree.ElementTree as ET
import os
import numpy as np
import cv2

# Class mapping for defect types
class_mapping = {
    'Background': 0,
    'Crack': 1,
    'Spallation': 2,
    'Efflorescence': 3,
    'ExposedBars': 4,
    'CorrosionStain': 5
}

def parse_codebrim_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    defect_data = []

    for defect in root.findall('Defect'):
        image_name = defect.attrib['name']
        labels = {}

        for defect_type in class_mapping.keys():
            value = int(defect.find(defect_type).text)
            labels[defect_type] = value

        defect_data.append((image_name, labels))
    
    return defect_data


def create_mask_from_labels(image_shape, labels, class_mapping):
    # Create an empty mask for each class
    mask = np.zeros((image_shape[0], image_shape[1], len(class_mapping) - 1), dtype=np.uint8)
    
    for defect, value in labels.items():
        if defect != 'Background' and value == 1:
            class_idx = class_mapping[defect] - 1
            mask[:, :, class_idx] = 1  # Set entire mask to 1 for this defect
    
    return mask


def process_codebrim_dataset(image_dir, xml_path, output_mask_dir):
    defect_data = parse_codebrim_xml(xml_path)

    for image_name, labels in defect_data:
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)

        if image is None:
            continue  # Skip if image not found

        mask = create_mask_from_labels(image.shape, labels, class_mapping)

        # Save mask
        mask_name = image_name.replace('.png', '_mask.png')
        mask_path = os.path.join(output_mask_dir, mask_name)
        cv2.imwrite(mask_path, mask * 255)  # Save as binary mask (255 for visibility)
        
        
class DataGenerator(Sequence):
    def __init__(self, image_paths, mask_paths, batch_size=8, image_size=(256, 256), num_classes=5):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self):
        return len(self.image_paths) // self.batch_size
    
    def __getitem__(self, idx):
        batch_images = self.image_paths[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_masks = self.mask_paths[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        images, masks = [], []

        for img_path, mask_path in zip(batch_images, batch_masks):
            image = cv2.imread(img_path)
            image = cv2.resize(image, self.image_size) / 255.0

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
            mask = np.expand_dims(mask, axis=-1)
            
            images.append(image)
            masks.append(mask)
        
        return np.array(images), np.array(masks)


def conv_block(x, num_filters, kernel_size=3, activation="relu", padding="same"):
    """
    A helper function to create a two-Conv layer block used repeatedly in the UNet encoder and decoder,
    with BatchNormalization and activation (ReLU as default).
    """
    x = Conv2D(num_filters, kernel_size, activation=None, padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    
    x = Conv2D(num_filters, kernel_size, activation=None, padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    
    return x

def unet_model(input_shape=(256, 256, 3), num_classes=5, base_filters=64, dropout_rate=0.1, max_pooling_shape=(2,2)):
    """
    Builds a UNet-like model for multi-class segmentation.
    :param input_shape: (height, width, channels)
    :param num_classes: number of classes for the output segmentation mask
    :param base_filters: number of filters in the first convolution block
    :return: Keras Model
    """
    
    inputs = Input(shape=input_shape)
    

    # Encoder (Downsampling)
    c1 = conv_block(inputs, base_filters)                    
    c1 = SpatialDropout2D(dropout_rate)(c1)
    p1 = MaxPooling2D(max_pooling_shape)(c1)
    
    c2 = conv_block(p1, base_filters * 2)
    c2 = SpatialDropout2D(dropout_rate)(c2)
    p2 = MaxPooling2D(max_pooling_shape)(c2)
    
    c3 = conv_block(p2, base_filters * 4)
    c3 = SpatialDropout2D(dropout_rate)(c3)                    
    p3 = MaxPooling2D(max_pooling_shape)(c3)
    
    c4 = conv_block(p3, base_filters * 8)
    c4 = SpatialDropout2D(dropout_rate)(c4)                    
    p4 = MaxPooling2D(max_pooling_shape)(c4)
    
    # Bridge
    c5 = conv_block(p4, base_filters * 16)                   
    
    # --------------------------
    # Decoder (Upsampling)
    # --------------------------
    # up 1
    u6 = Conv2DTranspose(base_filters * 8, (2, 2), strides=(2, 2), padding="same")(c5)
    u6 = concatenate([u6, c4])          # Skip connection
    c6 = conv_block(u6, base_filters * 8)
    
    # up 2
    u7 = Conv2DTranspose(base_filters * 4, (2, 2), strides=(2, 2), padding="same")(c6)
    u7 = concatenate([u7, c3])
    c7 = conv_block(u7, base_filters * 4)
    
    # up 3
    u8 = Conv2DTranspose(base_filters * 2, (2, 2), strides=(2, 2), padding="same")(c7)
    u8 = concatenate([u8, c2])
    c8 = conv_block(u8, base_filters * 2)
    
    # up 4
    u9 = Conv2DTranspose(base_filters, (2, 2), strides=(2, 2), padding="same")(c8)
    u9 = concatenate([u9, c1])
    c9 = conv_block(u9, base_filters)
    

    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

if __name__ == "__main__":
    model = unet_model(input_shape=(256, 256, 3), num_classes=5)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',  # or dice loss, focal loss, etc.
        metrics=['accuracy']
    )
    model.summary()


