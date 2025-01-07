import os
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
from numpy.typing import NDArray

import albumentations as A
import albumentations.core.bbox_utils as A_bbox_utils

from . import utils as u


class ImageAugumentor:
    """
    A class for augmenting images with bounding boxes and labels.
    Provides functionality for image preprocessing, augmentation, and format conversion.
    """

    def __init__(self, filepath: str,
                 label_values: List[Tuple[int, ...]],
                 bounding_boxes: NDArray[NDArray[int]]):
        """
        Initialize the ImageAugumentor with image file path, bounding boxes, and label values.

        Args:
            filepath (str): Path to the image file.
            bounding_boxes (NDArray[NDArray[int]]): List of bounding boxes in albumentations format.
            label_values (List[Tuple[int, ...]]): List of label values.

        Raises:
            IOError: If the provided filepath is not a valid image.
        """
        self._filepath = filepath
        self._input_img = self._load_input_image()
        self._label_values = label_values
        self._bounding_boxes = self._convert_to_albumentations_format(bounding_boxes)

        # Initialize output variables
        self._out_img = self._input_img
        self._out_img_name = None
        self._out_bboxes = self._bounding_boxes
        self._out_label_values = label_values

        # Apply initial padding
        self._add_padding()

    @staticmethod
    def _load_image(filepath) -> cv2.Mat:
        """
        Load and preprocess the input image.
        Returns:
            cv2.Mat: Preprocessed image in RGB format
        Raises:
            IOError: If the file is not a valid image
        """
        img = cv2.imread(filepath, 1)
        if img is None:
            raise IOError(f'{filepath} is not an image')
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _convert_to_albumentations_format(self, bounding_boxes: NDArray[NDArray[int]]) -> NDArray:
        """
        Convert bounding boxes to albumentations format.
        Args:
            bounding_boxes (NDArray[NDArray[int]]): Input bounding boxes
        Returns:
            NDArray: Bounding boxes in albumentations format
        """
        img_height, img_width = self._input_img.shape[:2]
        return A_bbox_utils.convert_bboxes_to_albumentations(
            bounding_boxes, 'pascal_voc', (img_height, img_width))

    @staticmethod
    def _create_bbox_params(min_visibility: float = 0) -> A.BboxParams:
        """
        Create bbox parameters for albumentation transformations.
        Args:
            min_visibility (float): Minimum visibility threshold for bboxes
        Returns:
            A.BboxParams: Configured parameters
        """
        return A.BboxParams(
            format='albumentations',
            label_fields=['label_fields'],
            min_visibility=min_visibility
        )

    def _get_augmentation_pipeline(self) -> A.Compose:
        """
        Create the augmentation pipeline with configured transformations.

        Returns:
            A.Compose: Configured augmentation pipeline
        """
        high_priority = [
            A.HorizontalFlip(p=0.5),
            A.PlasmaBrightnessContrast(p=0.2),
            A.VerticalFlip(p=0.5),
        ]

        return A.Compose([
            A.SomeOf(transforms=high_priority, n=np.random.randint(1, 4)),
        ], bbox_params=self._create_bbox_params())

    def _add_padding(self) -> 'ImageAugumentor':
        """
        Add padding to the image to make it square.

        Returns:
            ImageAugumentor: Self for method chaining
        """
        height, width = self._input_img.shape[:2]
        max_dim = max(width, height)

        pipeline = A.Compose([
            A.PadIfNeeded(max_dim, max_dim, border_mode=cv2.BORDER_CONSTANT)
        ], bbox_params=self._create_bbox_params())

        augmented = pipeline(
            image=self._input_img,
            bboxes=self._bounding_boxes,
            label_fields=self._label_values
        )

        self._update_augmented_outputs(augmented)
        return self

    def _update_augmented_outputs(self, augmented: Dict[str, Any]) -> None:
        """
        Update internal state with augmented outputs.

        Args:
            augmented (Dict[str, Any]): Augmentation results
        """
        self._out_img = augmented['image']
        self._out_bboxes = augmented['bboxes']
        self._out_label_values = augmented['label_fields']

    def resize(self, out_width: int = 512, out_height: int = 512) -> 'ImageAugumentor':
        """
        Resize the image to the specified width and height.

        Args:
            out_width (int, optional): Output width. Defaults to 512.
            out_height (int, optional): Output height. Defaults to 512.

        Returns:
            self: The instance itself.
        """
        bboxparams = self._create_bbox_params(min_visibility=0)
        pipeline = A.Compose([
            A.Resize(width=out_width, height=out_height)
        ], bbox_params=bboxparams)

        augumented_img = pipeline(image=self._out_img, bboxes=self._out_bboxes, label_fields=self._label_values)
        self._out_img = augumented_img['image']
        self._bounding_boxes = augumented_img['bboxes']
        self._out_label_values = augumented_img['label_fields']
        return self

    def apply_augumentations(self) -> 'ImageAugumentor':
        """
        Apply a series of augmentations to the image.

        Returns:
            ImageAugumentor: Self for method chaining
        """
        pipeline = self._get_augmentation_pipeline()
        augmented = pipeline(
            image=self._out_img,
            bboxes=self._out_bboxes,
            label_fields=self._label_values
        )
        self._update_augmented_outputs(augmented)
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
    def processed_bboxes_albumentations(self) -> list[NDArray[int]]:
        """
        Get the processed bounding boxes.

        Returns:
            list[tuple[int]]: The processed bounding boxes in albumentations format.
        """
        return self._out_bboxes.tolist()

    @property
    def processed_bboxes_pascal_voc(self) -> list[tuple[int]]:
        height = self.processed_image.shape[0]
        width = self.processed_image.shape[1]
        pascal_voc_bboxes = A_bbox_utils.convert_bboxes_from_albumentations(
                                        self._out_bboxes, 'pascal_voc',
                                        shape=(height, width))
        return pascal_voc_bboxes.tolist()

    @property
    def processed_bboxes_yolo(self) -> list[list[int]]:
        height = self.processed_image.shape[0]
        width = self.processed_image.shape[1]
        yolo_bboxes = A_bbox_utils.convert_bboxes_from_albumentations(self._out_bboxes, 'yolo',
                                                                      (height, width))
        return yolo_bboxes.tolist()

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
            'bboxes': self.processed_bboxes_yolo,
            'label_values': self.processed_label_values
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