import os

import cv2
import pandas as pd
import numpy as np
from skmultilearn.model_selection import iterative_train_test_split

from src import constants as c
import src.image_processing as improc
import src.utils as u

def augment_image_and_save_to_folder(input_file_path: str, output_folder: str, data: pd.DataFrame) -> dict:
    bboxes = data[c.bbox_coordinate_names].astype(int).to_numpy()
    multi_hot_encoded_labels = [list(x) for x in data[c.defect_names].astype(int).to_numpy()]

    # Generate augmented image variations
    augmented_images = improc.generate_augmented_images(
        image_path=input_file_path,
        bounding_boxes=bboxes,
        label_values=multi_hot_encoded_labels,
        resize=True,
        target_width=512,
        target_height=512
    )

    # Save each augmented image and collect metadata
    image_metadata = []
    for aug_img in augmented_images:
        output_path = os.path.join(output_folder, aug_img.processed_image_name)
        cv2.imwrite(output_path, aug_img.processed_image)
        image_metadata.append(aug_img.metadata_dict)

    # Combine metadata from all images into single dictionary
    return u.unpack_lists(
            list_of_dicts=image_metadata,
            col_list=image_metadata[0].keys()
        )


def train_test_val_image_split(input_df: pd.DataFrame,
                               test_size: float = 0.2,
                               val_size: float = 0.1) -> dict[str, np.ndarray]:
    """
    Splits the input DataFrame into training, testing, and validation sets
    based on images. If there are multiple entries of the same image, it merges them by
    maximum value of each label.

    Parameters:
    ----------
    input_df : pd.DataFrame
        A pandas DataFrame containing image file names, bounding box coordinates
        in format (xmin, xmax, ymin, ymax) and their corresponding multi-hot encoded labels.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    val_size : float, default=0.1
        Proportion of the dataset to include in the validation split.

    Returns:
    -------
    dict[str, np.ndarray]
        A dictionary with keys 'train', 'test', and 'val',
        each containing a NumPy array of image file names for the respective dataset.
    """
    if input_df.empty:
        raise ValueError('Input DataFrame is empty')

    if not (0 < test_size < 1) or not (0 <= val_size < 1):
        raise ValueError("test_size and val_size must be between 0 and 1.")

    if (test_size + val_size) >= 1:
        raise ValueError("The sum of test_size and val_size must be less than 1.")

    # Drop bounding box coordinates and group by image
    df = input_df.drop(['xmin', 'xmax', 'ymin', 'ymax'], axis=1)
    df = df.groupby('image', as_index=False).max()

    X = df['image'].to_numpy()
    y = df.drop('image', axis=1).to_numpy()

    # First split: separate training from combined test+val
    X_train, y_train, X_rem, y_rem = iterative_train_test_split(
        X.reshape(-1, 1), y, test_size=(test_size + val_size))

    # Second split: separate test and validation from the remainder
    X_test, y_test, X_val, y_val = iterative_train_test_split(
        X_rem, y_rem, test_size=(val_size / (test_size + val_size)))

    return {
        'train': X_train.ravel(),  # Convert 2D array back to 1D
        'test': X_test.ravel(),
        'val': X_val.ravel()
    }