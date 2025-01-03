import os

import numpy as np
import pandas as pd
from . import LabelParser
from keras.api.utils import Sequence
import albumentations as A
import cv2

def build_augmentation_pipeline():
    return A.Compose([
        A.Rotate(limit=30, p=0.7),  # Rotate by ±30 degrees
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.HorizontalFlip(p=0.5)
    ],
    bbox_params=A.BboxParams(  # Add bbox_params here
        format='pascal_voc',  # or 'yolo', 'coco', etc.
        label_fields=['category_ids'],  # Match with the label field used in augmentation
        min_area=0.0,
        min_visibility=0.3,  # Ignore bboxes with low visibility
        check_each_transform=True
    ))



# Custom DataGenerator that takes paths dynamically
class DataGenerator(Sequence):
    def __init__(self, annotations_path, images_path, batch_size, img_size, augment=False):
        super(DataGenerator, self).__init__()  # Call to superclass __init__
        self.annotations_path = annotations_path
        self.images_path = images_path
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.augment_pipeline = build_augmentation_pipeline()
        self.annotations_parser = LabelParser(self.annotations_path)
        self.image_ids = self.annotations_parser.annotations_df['img_name'].unique()
        np.random.shuffle(self.image_ids)

    def __len__(self):
        return int(np.floor(len(self.image_ids) / self.batch_size))

    def __getitem__(self, index):
        batch_ids = self.image_ids[index * self.batch_size:(index + 1) * self.batch_size]
        batch_imgs, batch_labels = self.__data_generation(batch_ids)
        return np.array(batch_imgs), np.array(batch_labels)

    def __data_generation(self, batch_ids):
        batch_imgs = []
        batch_labels = []

        for img_name in batch_ids:
            img_path = os.path.join(self.images_path, img_name)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Get annotations for this image
            annotations = self.annotations_parser.annotations_df[
                self.annotations_parser.annotations_df['img_name'] == img_name
                ]

            # Extract bounding boxes and multi-hot labels
            bboxes = annotations[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
            labels = [tuple(row) for row in annotations[
                ['Background', 'Crack', 'Spallation', 'Efflorescence', 'ExposedBars', 'CorrosionStain']
            ].values.tolist()]

            # If no bounding boxes exist, treat the image as "background"
            if len(bboxes) == 0:
                bboxes = [[0, 0, image.shape[1], image.shape[0]]]  # Full-image bbox
                labels = [tuple([1, 0, 0, 0, 0, 0])]  # Background-only label

            # Perform augmentation with bbox and labels
            if self.augment:
                augmented = self.augment_pipeline(
                    image=image,
                    bboxes=bboxes,
                    category_ids=labels
                )

                # Filter out invalid bounding boxes
                valid_bboxes = []
                valid_labels = []
                for bbox, label in zip(augmented['bboxes'], augmented['category_ids']):
                    x_min, y_min, x_max, y_max = bbox
                    if (x_max > x_min) and (y_max > y_min):
                        valid_bboxes.append(bbox)
                        valid_labels.append(label)

                image = augmented['image']
                bboxes = valid_bboxes
                labels = valid_labels

            image = cv2.resize(image, self.img_size)
            batch_imgs.append(image)

            # If no valid bounding boxes remain, label as background
            if len(labels) == 0:
                batch_labels.append(tuple([1, 0, 0, 0, 0, 0]))  # Background-only
            else:
                batch_labels.append(labels[0])

        return batch_imgs, batch_labels

    def on_epoch_end(self):
        np.random.shuffle(self.image_ids)
