import tensorflow as tf
import keras.api as keras
from src.annotation_extractor import xml_annotations_to_dataframe

xmls_folder = r'kaggle/input/codebrim-original/original_dataset/annotations/'
imgs_folder = r'kaggle/input/codebrim-original/original_dataset/images/'
df = xml_annotations_to_dataframe(xmls_folder)

# Normalize bounding boxes based on image dimensions
df['xmax'] = df['xmin'] + df['width']
df['ymax'] = df['ymin'] + df['height']

# Normalize box coordinates to [0,1]
df['xmin'] = df['xmin'] / df['width']
df['ymin'] = df['ymin'] / df['height']
df['xmax'] = df['xmax'] / df['width']
df['ymax'] = df['ymax'] / df['height']

# Defect labels for multi-hot encoding
defect_labels = ['Background', 'Crack', 'Spallation', 'Efflorescence', 'ExposedBars', 'CorrosionStain']

# Prepare bounding box data without merging overlapping boxes
bbox_labels = df[['img', 'xmin', 'ymin', 'xmax', 'ymax'] + defect_labels]

# Prepare image paths, bounding boxes, and labels
img_names = bbox_labels['img'].values
bounding_boxes = bbox_labels[['xmin', 'ymin', 'xmax', 'ymax']].values
multi_hot_labels = bbox_labels[defect_labels].values

print(bounding_boxes.shape)  # Should be (num_samples, 4)
print(multi_hot_labels.shape)  # Should be (num_samples, 6)

bounding_boxes = bounding_boxes.reshape(-1, 4).astype('float32')  # (num_samples, 4)
multi_hot_labels = multi_hot_labels.astype('float32')  # (num_samples, 6)

print(bounding_boxes.shape)  # Should be (num_samples, 4)
print(multi_hot_labels.shape)  # Should be (num_samples, 6)

def preprocess_data(file_path, bbox, label):
    img = tf.io.read_file(imgs_folder + file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = img / 255.0
    return img, (bbox, label)  # Return bbox and label as a tuple


# Create TensorFlow Dataset with images, bounding boxes, and labels
def load_dataset(image_paths, inp_bboxes, labels, batch_size=32):
    bboxes = inp_bboxes.reshape(-1, 4).astype('float32')  # Adjust shape to (num_samples, 4)
    labels = labels.astype('float32')  # Ensure labels are float32

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, bboxes, labels))
    dataset = dataset.map(preprocess_data)

    padded_shapes = ([224, 224, 3], ([4], [len(defect_labels)]))  # Tuple of shapes
    padding_values = (tf.constant(0.0, dtype=tf.float32),  # Image padding
                      (tf.constant(0.0, dtype=tf.float32),  # Bounding box padding
                       tf.constant(0.0, dtype=tf.float32)))  # Label padding

    dataset = dataset.padded_batch(batch_size,
                                   padded_shapes=padded_shapes,
                                   padding_values=padding_values)
    return dataset



# Model with two outputs: bounding box regression and multi-label classification
input_img = keras.layers.Input(shape=(224, 224, 3))
x = keras.layers.Conv2D(32, (3, 3), activation='relu')(input_img)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(128, activation='relu')(x)

bbox_output = keras.layers.Dense(4, name='boxes')(x)  # Bounding box coordinates
label_output = keras.layers.Dense(len(defect_labels), activation='sigmoid', name='labels')(x)  # Multi-hot labels


model = keras.models.Model(inputs=input_img, outputs=[bbox_output, label_output])

if __name__ == '__main__':
    # Build the dataset
    train_dataset = load_dataset(img_names, bounding_boxes, multi_hot_labels)

    # Compile the model
    model.compile(
        optimizer='adam',
        loss={
            'boxes': 'mse',  # MSE for bounding box regression
            'labels': 'binary_crossentropy'  # Binary cross-entropy for multi-label classification
        },
        metrics={
            'boxes': 'mse',
            'labels': 'accuracy'
        }
    )

    # Train the model
    model.fit(train_dataset, epochs=10)
