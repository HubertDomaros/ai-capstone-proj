import os
from enum import Enum
from dataclasses import dataclass

import cv2
import pandas as pd
import numpy as np
import numpy.typing as npt
import albumentations as A
import matplotlib.pyplot as plt


from . import constants, utils
from . import constants as consts


class ImageAugumentor:
    def __init__(self, filepath: str, bounding_boxes: list[list[int]], 
                 label_names: list[str] = [],
                 label_values: tuple|None = None,
                 bbox_coord_format: str = 'pascal_voc'):
        
        self._filepath = filepath
        
        self._input_img: cv2.Mat = cv2.imread(filepath, 1)
        
        if self._input_img is None:
            raise IOError(f'{filepath} is not an image')
        
        self._input_img: cv2.Mat = cv2.cvtColor(self._input_img, cv2.COLOR_BGR2RGB)
        
        
        self._bounding_boxes = bounding_boxes
        self._label_names = label_names
        self._label_values = label_values
        self._bbox_coord_format = bbox_coord_format
        
        self._out_img = self._input_img
        self._out_bboxes = self._bounding_boxes
        
        self._add_padding()
    
    def _add_padding(self) -> dict:
        width = self._input_img.shape[0]
        height = self._input_img.shape[1]
        
        pipeline = 0
        
        bboxparams = A.BboxParams(format=self._bbox_coord_format, label_fields=['label_fields'])
            
        if width > height:
            pipeline = A.Compose([
                A.PadIfNeeded(width, width, border_mode=cv2.BORDER_CONSTANT)
            ], bbox_params=bboxparams)
        else:
            pipeline = A.Compose([
                A.PadIfNeeded(width, width, border_mode=cv2.BORDER_CONSTANT)
            ], bbox_params=bboxparams)
                
        augumented_img = pipeline(image = self._input_img, bboxes = self._bounding_boxes, 
                                    label_fields=self._label_values)
        
        self._out_img = augumented_img['image']
        self._out_bboxes = augumented_img['bboxes']
        
        return self
    
    def resize(self, out_width=512, out_height=512):
        bboxparams = A.BboxParams(format=self._bbox_coord_format, label_fields=['label_fields'])
        
        pipeline = A.Compose([
                A.Resize(width=out_width, height=out_height, border_mode=cv2.BORDER_CONSTANT)
            ], bbox_params=bboxparams)
        
        augumented_img = pipeline(image=self._out_img, bboxes = self._out_bboxes)
        
        self._out_img = augumented_img['image']
        self._bounding_boxes = augumented_img['bboxes']
        
        return self
    
    def apply_augumentations(self):
        bboxparams = A.BboxParams(format=self._bbox_coord_format, label_fields=['label_fields'])
        
        high_priority = [
            A.Rotate(limit=30, p=0.7),
            A.HorizontalFlip(p=0.5),  
            A.PlasmaBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
            A.Perspective(p=0.5)
        ]
        
        medium_priority = [
            A.ElasticTransform(alpha=10, sigma=20, p=0.25),
            A.Affine(scale=np.random.uniform(0.8, 1), p=0.4),
            A.Affine(translate_percent=np.random.randint(5, 10), p=0.4), 
            A.VerticalFlip(p=0.5),
            
            A.GaussianBlur(blur_limit=(3, 7), p=0.45), 
            A.GaussNoise(p=0.35)
        ]
            
        pipeline = A.Compose([
            A.SomeOf(transforms=high_priority, n=np.random.randint(1, 3)),
            A.RandomOrder(transforms=medium_priority, n=np.random(1, 3))
        ], bbox_params=bboxparams)
        
        augumented_img = pipeline(image = self._out_img, bboxes = self._out_bboxes, 
                                    label_fields=self._label_values)
        
        self._out_img = augumented_img['image']
        self._out_bboxes = augumented_img['bboxes']
    
        return self
    
    def plot_processed_img_with_bboxes(self) -> None:
        fig, ax = plt.subplots()
        ax.imshow(self._out_img)
        for bbox in self._out_bboxes:
            i = 0
            bbox_rectangle = utils.draw_bounding_box(bbox[0], bbox[1], bbox[2], bbox[3], 
                                                     constants.colors[i], 2)
            ax.add_patch(bbox_rectangle)
            i+=1
        
        plt.imshow(self._out_img)
        plt.show()
    
    @property
    def processed_image(self) -> cv2.typing.MatLike:
        return self._out_img
    
    @property
    def processed_bboxes(self) -> npt.ArrayLike[npt.ArrayLike[int]]:
        return self._out_bboxes


@dataclass
class ImageData:
     images: cv2.typing.MatLike
     image_name: str
     bboxes_list: list[list[int]]
     label_values_list: list[list[int]]
        
    
def generate_augumented_images(image_path: str, 
                               bounding_boxes: list[str], label_values: list[list[int]],
                               n: int=8, resize: bool = True,
                               out_width: int = 512, out_height: int = 512) -> list[ImageData]:
    
    
    image_name_wo_ext = os.path.splitext(os.path.basename(image_path))[0]
    file_extension = os.path.splitext(os.path.basename(image_path))[1]
    
    images: list[ImageData] = []
    
    for i in range(0, n):
        image_data = ImageData()
        try:
            aug_obj = ImageAugumentor(image_path, bounding_boxes=bounding_boxes, label_values=label_values)
        except Exception as e:
            print(e, 'skipping file processing')
            
        if resize:
            aug_obj.resize(out_width=out_width, out_height=out_height)
            
        aug_obj.apply_augumentations()
        
        image_data.image = aug_obj.processed_image
        image_data.image_name = f'{image_name_wo_ext}_{i}.{file_extension}'
        
        
        for bbox in aug_obj.processed_bboxes:
            image_data.bboxes_list.append(bbox)
        for labels in label_values:
            image_data.label_values_list.append(labels)
        
        
    return images


def process_images_in_folder(folder_path: str, input_df: pd.DataFrame):
    file_set = set(os.listdir(folder_path))

    for filename, filedata in input_df.groupby[consts.IMG]:
        