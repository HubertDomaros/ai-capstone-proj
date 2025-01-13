import os
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.ops import box_convert
from PIL import Image


class ImageAugmentorGPU:
    """
    A GPU-accelerated class for augmenting images with bounding boxes and labels.
    Provides functionality for image preprocessing, augmentation, and format conversion.
    """

    def __init__(self, filepath: str,
                 label_values: List[Tuple[int, ...]],
                 bounding_boxes: np.ndarray,
                 device: Optional[torch.device] = None):
        """
        Initialize the ImageAugmentorGPU with image file path, bounding boxes, and label values.

        Args:
            filepath (str): Path to the image file
            label_values (List[Tuple[int, ...]]): List of label values
            bounding_boxes (np.ndarray): Bounding boxes in Pascal VOC format [x1, y1, x2, y2]
            device (torch.device, optional): Device to use for computation
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load and convert image to tensor
        self._filepath = filepath
        self._original_img = self._load_image(filepath)
        self._original_label_values = label_values
        self._original_bboxes = self._convert_boxes_to_tensor(bounding_boxes)

        # Initialize output variables
        self._augmented_img = self._original_img
        self._augmented_img_name = None
        self._augmented_bboxes = self._original_bboxes
        self._augmented_label_values = label_values

        # Apply initial padding
        self._add_padding()

    def _load_image(self, filepath: str) -> torch.Tensor:
        """
        Load and preprocess the input image.

        Returns:
            torch.Tensor: Preprocessed image tensor in RGB format [C, H, W]
        """
        img = cv2.imread(filepath, 1)
        if img is None:
            raise IOError(f'{filepath} is not an image')

        # Convert BGR to RGB and normalize to [0, 1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        return img_tensor.to(self.device)

    def _convert_boxes_to_tensor(self, boxes: np.ndarray) -> torch.Tensor:
        """
        Convert numpy boxes to normalized tensor format.

        Args:
            boxes: Bounding boxes in Pascal VOC format [x1, y1, x2, y2]
        """
        boxes_tensor = torch.from_numpy(boxes).float()
        img_size = torch.tensor([self._original_img.shape[2],
                                 self._original_img.shape[1]]).to(self.device)

        # Normalize coordinates
        boxes_tensor[:, [0, 2]] /= img_size[0]
        boxes_tensor[:, [1, 3]] /= img_size[1]
        return boxes_tensor.to(self.device)

    def _add_padding(self) -> 'ImageAugmentorGPU':
        """Add padding to make the image square."""
        _, h, w = self._augmented_img.shape
        max_dim = max(h, w)

        # Calculate padding
        pad_h = max_dim - h
        pad_w = max_dim - w
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        # Apply padding
        self._augmented_img = F.pad(
            self._augmented_img.unsqueeze(0),
            (pad_left, pad_right, pad_top, pad_bottom),
            mode='constant',
            value=0
        ).squeeze(0)

        # Adjust bounding boxes for padding
        if pad_left > 0 or pad_top > 0:
            self._augmented_bboxes[:, [0, 2]] = (self._augmented_bboxes[:, [0, 2]] * w + pad_left) / max_dim
            self._augmented_bboxes[:, [1, 3]] = (self._augmented_bboxes[:, [1, 3]] * h + pad_top) / max_dim

        return self

    def resize(self, out_width: int = 512, out_height: int = 512) -> 'ImageAugmentorGPU':
        """Resize the image to specified dimensions."""
        # Resize image
        self._augmented_img = F.interpolate(
            self._augmented_img.unsqueeze(0),
            size=(out_height, out_width),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        return self

    def apply_augmentations(self) -> 'ImageAugmentorGPU':
        """Apply augmentations to the image and boxes."""
        # Random horizontal flip
        if torch.rand(1).item() > 0.5:
            self._augmented_img = TF.hflip(self._augmented_img)
            self._augmented_bboxes[:, [0, 2]] = 1 - self._augmented_bboxes[:, [2, 0]]

        # Random vertical flip
        if torch.rand(1).item() > 0.5:
            self._augmented_img = TF.vflip(self._augmented_img)
            self._augmented_bboxes[:, [1, 3]] = 1 - self._augmented_bboxes[:, [3, 1]]

        # Adjust brightness and contrast
        if torch.rand(1).item() > 0.8:
            brightness_factor = torch.rand(1).item() * 0.4 + 0.8  # 0.8 to 1.2
            contrast_factor = torch.rand(1).item() * 0.4 + 0.8  # 0.8 to 1.2
            self._augmented_img = TF.adjust_brightness(self._augmented_img, brightness_factor)
            self._augmented_img = TF.adjust_contrast(self._augmented_img, contrast_factor)

        return self

    @property
    def processed_image(self) -> torch.Tensor:
        """Get the augmented image tensor."""
        return self._augmented_img

    @property
    def processed_image_numpy(self) -> np.ndarray:
        """Get the augmented image as numpy array."""
        img = (self._augmented_img.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    @property
    def processed_bboxes_pascal_voc(self) -> List[List[float]]:
        """Get bounding boxes in Pascal VOC format."""
        h, w = self._augmented_img.shape[1:]
        boxes = self._augmented_bboxes.clone()
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h
        return boxes.cpu().numpy().tolist()

    @property
    def processed_bboxes_yolo(self) -> List[List[float]]:
        """Get bounding boxes in YOLO format [x_center, y_center, img_width, img_height]."""
        boxes = self._augmented_bboxes.clone()
        # Convert from x1,y1,x2,y2 to x_center,y_center,img_width,img_height
        boxes = box_convert(boxes, 'xyxy', 'cxcywh')
        return boxes.cpu().numpy().tolist()

    @property
    def processed_label_values(self) -> List[Tuple[int, ...]]:
        """Get the processed label values."""
        return self._augmented_label_values

    @property
    def metadata_dict(self) -> Dict:
        """Generate metadata dictionary for the processed image and boxes."""
        return {
            'img': [self._augmented_img_name] * len(self._augmented_bboxes),
            'bboxes': self.processed_bboxes_yolo,
            'label_values': self.processed_label_values
        }