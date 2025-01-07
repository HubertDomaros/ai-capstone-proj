import os
from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import albumentations as A
import albumentations.core.bbox_utils as A_bbox_utils

from . import utils as u
from . import constants as c

class ImageAugumentor:
    """
    A class for augmenting images with bounding boxes and labels.
    """

    def __init__(self, filepath: str,
                 label_values: list[tuple[int, ...]],
                 bounding_boxes: NDArray[NDArray[int]]):
        """
        Initialize the ImageAugumentor with image file path, bounding boxes, and label values.

        Args:
            filepath (str): Path to the image file.
            bounding_boxes (NDArray[NDArray[int]]): List of bounding boxes in albumentations format.
            label_values (list[tuple[int, ...]]): List of label values.
        """
        self._filepath = filepath
        self._input_img: cv2.Mat = cv2.imread(filepath, 1)
        if self._input_img is None:
            raise IOError(f'{filepath} is not an image')
        self._input_img: cv2.Mat = cv2.cvtColor(self._input_img, cv2.COLOR_BGR2RGB)
        
        self._label_values = label_values
        img_height = self._input_img.shape[0]
        img_width = self._input_img.shape[1]
        self._bounding_boxes = A_bbox_utils.convert_bboxes_to_albumentations(bounding_boxes, 'pascal_voc',
                                                                                  (img_height, img_width))

        self._out_img = self._input_img
        self._out_img_name = None
        self._out_bboxes = self._bounding_boxes
        self._out_label_values = label_values
        self._add_padding()

    def _add_padding(self):
        """
        Add padding to the image to make it square.
        """
        height = self._input_img.shape[0]
        width = self._input_img.shape[1]
        bboxparams = A.BboxParams(format='albumentations', label_fields=['label_fields'])

        if width > height:
            pipeline = A.Compose([
                A.PadIfNeeded(width, width, border_mode=cv2.BORDER_CONSTANT)
            ], bbox_params=bboxparams)
        else:
            pipeline = A.Compose([
                A.PadIfNeeded(width, width, border_mode=cv2.BORDER_CONSTANT)
            ], bbox_params=bboxparams)

        augumented_img = pipeline(image=self._input_img, bboxes=self._bounding_boxes,
                                  label_fields=self._label_values)

        self._out_img = augumented_img['image']
        self._out_bboxes = augumented_img['bboxes']
        #self._out_bboxes = [[max(0.001, min(0.9999, x)) for x in box] for box in self._out_bboxes]
        self._out_label_values = augumented_img['label_fields']
        return self

    def resize(self, out_width=512, out_height=512):
        """
        Resize the image to the specified width and height.

        Args:
            out_width (int, optional): Output width. Defaults to 512.
            out_height (int, optional): Output height. Defaults to 512.

        Returns:
            self: The instance itself.
        """
        bboxparams = A.BboxParams(format='albumentations', label_fields=['label_fields'])
        pipeline = A.Compose([
            A.Resize(width=out_width, height=out_height)
        ], bbox_params=bboxparams)

        augumented_img = pipeline(image=self._out_img, bboxes=self._out_bboxes, label_fields=self._label_values)
        self._out_img = augumented_img['image']
        self._bounding_boxes = augumented_img['bboxes']
        self._out_label_values = augumented_img['label_fields']
        return self

    def apply_augumentations(self):
        """
        Apply a series of augmentations to the image.

        Returns:
            self: The instance itself.
        """
        bboxparams = A.BboxParams(format='albumentations',
                                  label_fields=['label_fields'],
                                  min_visibility=0)

        high_priority = [
            #A.Rotate(limit=15, p=0.7, border_mode=cv2.BORDER_CONSTANT),
            A.HorizontalFlip(p=0.5),
            A.PlasmaBrightnessContrast(p=0.8),
            A.Perspective(p=0.5)
        ]
        medium_priority = [
            A.ElasticTransform(alpha=10, sigma=20, p=0.25),
            A.VerticalFlip(p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.45),
            A.GaussNoise(p=0.35, std_range=(0.1, 0.2)),
            A.ChannelShuffle(p=0.2),
            A.ShiftScaleRotate(rotate_limit=(0, 0))
        ]
        pipeline = A.Compose([
            A.SomeOf(transforms=high_priority, n=np.random.randint(1, 4)),
            A.RandomOrder(transforms=medium_priority, n=np.random.randint(1, 3))
        ], bbox_params=bboxparams)

        augumented_img = pipeline(image=self._out_img, bboxes=self._out_bboxes,
                                  label_fields=self._label_values)

        self._out_img = augumented_img['image']
        self._out_bboxes = augumented_img['bboxes']
        self._out_label_values = augumented_img['label_fields']
        return self

    def plot_processed_img_with_bboxes(self) -> None:
        """
        Plot the processed image with bounding boxes.
        """
        u.plot_img_with_bboxes(self._out_img, self.processed_bboxes_pascal_voc)

    @property
    def processed_image(self) -> cv2.typing.MatLike:
        """
        Get the processed image.

        Returns:
            cv2.typing.MatLike: The processed image.
        """
        return self._out_img
    
    
    @property
    def processed_image_name(self) -> str:
        if self._out_img_name is None:
            raise ValueError('Image name is not set. Set image name first '+
                             'by calling setter processsed_image_name()')
        return self._out_img_name
    
    @processed_image_name.setter
    def processed_image_name(self, img_name: str) -> None:
        self._out_img_name = img_name


    @property
    def processed_bboxes_albumentations(self) -> list[tuple[int]]:
        """
        Get the processed bounding boxes.

        Returns:
            list[tuple[int]]: The processed bounding boxes in albumentations format.
        """
        return self._out_bboxes

    @property
    def processed_bboxes_pascal_voc(self) -> list[tuple[int]]:
        pascal_voc_bboxes = A_bbox_utils.convert_bboxes_from_albumentations(
                                        self._out_bboxes, 'pascal_voc',
                                        shape=self.processed_image.shape)
        return pascal_voc_bboxes.tolist()

    @property
    def processed_label_values(self) -> list[tuple[int, ...]]:
        return self._out_label_values

    @property
    def metadata_dict(self) -> dict:
        """
        Generate a dictionary containing metadata for the processed image, bounding boxes, and label values.

        Returns:
            dict: A dictionary with keys 'img', 'bboxes', and 'label_values'.
        """
        return {
            'img': [self.processed_image_name] * len(self._out_bboxes),
            'bboxes': self._out_bboxes,
            'label_values': self._out_label_values
        }
    

def generate_augmented_images(image_path: str,
                              bounding_boxes: NDArray[NDArray[int]],
                              label_values: list[list[int]],
                              n: int = 8, resize: bool = True,
                              out_width: int = 512, out_height: int = 512) -> list[ImageAugumentor]:
    """
    Generate augmented images from the given image path, bounding boxes, and label values.

    Args:
        image_path (str): Path to the image file.
        bounding_boxes (NDArray[NDArray[int]]): List of bounding boxes in albumentations format.
        label_values (list[list[int]]): List of label values in multi-hot encoding format.
        n (int, optional): Number of augmented images to generate. Defaults to 8.
        resize (bool, optional): Whether to resize the images. Defaults to True.
        out_width (int, optional): Output width for resizing. Defaults to 512.
        out_height (int, optional): Output height for resizing. Defaults to 512.

    Returns:
        list[ImageAugumentor]: List of augmented image data.
    """
    image_name_wo_ext = os.path.splitext(os.path.basename(image_path))[0]
    file_extension = os.path.splitext(os.path.basename(image_path))[1]
    images: list[ImageAugumentor] = []
    label_values: list[tuple[int, ...]] = [tuple(i) for i in label_values]

    aug_img = 0
    for i in range(0, n):
        try:
            aug_obj = ImageAugumentor(image_path, bounding_boxes=bounding_boxes, label_values=label_values)
        except Exception as e:
            print(e, 'skipping file processing')
            continue

        if resize:
            aug_obj.resize(out_width=out_width, out_height=out_height)

        aug_obj.apply_augumentations()
        aug_obj.processed_image_name = f'{image_name_wo_ext}_{i+1}{file_extension}'
        
        images.append(aug_obj)
        
    return images

        
    
    
# def process_images_in_folder(folder_path: str, input_df: pd.DataFrame):
#     file_set = set(os.listdir(folder_path))

#     for filename, filedata in input_df.groupby[c.IMG]:
        