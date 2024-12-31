import json
from math import e
import os
import shutil

import pandas as pd
import numpy as np
from skmultilearn.model_selection import iterative_train_test_split

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

    Returns:
    -------
    dict[str, ndarray]
        A dictionary with keys 'train', 'test', and 'val',
        each containing a NumPy array of image file names for the respective dataset.
    """

    
    df = input_df.drop(['xmin', 'xmax', 'ymin', 'ymax'], axis=1)
    df = df.groupby('image', as_index=False).max()
    
    X: np.ndarray = df['image'].to_numpy()
    y: np.ndarray = df.drop('image', axis=1).to_numpy()

    X_train, _, X_rem, y_rem = iterative_train_test_split(X, y, test_size=(test_size + val_size))

    X_test, _, X_val, _ = iterative_train_test_split(X_rem, y_rem, 
                                                     test_size=(val_size / (test_size + val_size)))

    return {
        'train': X_train,
        'test': X_test,
        'val': X_val,       
    }


def dict_to_json(input_dict: dict[str, np.ndarray], out_filepath: str) -> None:
    """
    Saves a dictionary to a JSON file.

    Parameters:
    ----------
    input_dict : dict[str, np.ndarray]
        The dictionary to save.
    out_filepath : str
        The path to the output JSON file.
    """
    with open(out_filepath, 'w') as file:
        json.dump(input_dict, file)


def put_splited_images_in_folders(json_filepath: str, input_dir: str, out_dir: str) -> None:
    with open(json_filepath, 'r') as json_file:
        json_obj: dict = json.load(json_file)
        datasets = ['train', 'test', 'val']

    for dataset_name in datasets:
        dataset_dir = os.path.join(out_dir, dataset_name)
        if os.path.exists(out_dir):
            raise OSError(f'Cannot make directory, {dataset_dir} already exists')
        
        os.makedirs(dataset_dir)
        print(f"Created directory {dataset_dir}")

        for img in json_obj[dataset_name]:
            src = os.path.join(input_dir, img)
            out = os.path.join(dataset_dir, img)
            
            try:
                raise_exeption_if_file_exists(out)
            except FileExistsError:
                print(f'File {img} in path {dataset_dir} already exists. Skipping copying operation')
                continue

            os.symlink(src, out)


def raise_exeption_if_file_exists(filepath: str):
    if os.path.exists(filepath):
        raise FileExistsError