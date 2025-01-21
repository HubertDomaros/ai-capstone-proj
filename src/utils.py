from typing import LiteralString

import matplotlib.pyplot as plt
import xmltodict
from skmultilearn.model_selection import iterative_train_test_split
from tqdm import tqdm

from . import constants as c
import pandas as pd
import numpy as np
import json
import os
from os import listdir
from os.path import join
import shutil
from shutil import copyfile


def draw_bounding_box(xmin: int, ymin: int, xmax: int, ymax: int, edge_color: str = 'blue', linewidth: int = 2):
    """
    Create a rectangular bounding box for visualization.
    Args:
        xmin (int): Minimum x-coordinate of the bounding box
        ymin (int): Minimum y-coordinate of the bounding box
        xmax (int): Maximum x-coordinate of the bounding box
        ymax (int): Maximum y-coordinate of the bounding box
        edge_color (str, optional): Color of the bounding box edge. Defaults to 'blue'
        linewidth (int, optional): Width of the bounding box line. Defaults to 2
    Returns:
        plt.Rectangle: A matplotlib Rectangle patch representing the bounding box
    """
    width = xmax - xmin
    height = ymax - ymin
    x = xmin
    y = ymin
    return plt.Rectangle((x, y), width=width, height=height,
                        edgecolor=edge_color, facecolor='none', linewidth=linewidth)


def unpack_lists(list_of_dicts, col_list) -> dict[str, list[str | int]]:
    """
    Unpack a list of dictionaries into a single dictionary with lists as values.
    Handles numpy arrays by converting them to lists.
    Args:
        list_of_dicts (List[dict]): List of dictionaries to unpack
        col_list (List[str]): List of column names to include in output
    Returns:
        dict[str, list[str|int]]: Dictionary with column names as keys and lists of values
    """

    out_dict = {}
    for key in col_list:
        out_dict[key] = []

    # Process each dictionary in the input list
    for single_dict in list_of_dicts:
        for key in single_dict.keys():
            val = single_dict[key]
            if isinstance(val, np.ndarray):
                val = val.tolist()
            out_dict[key].extend(val)

    return out_dict


def list_of_dicts_to_dataframe(list_of_dicts, columns_list) -> pd.DataFrame:
    """
    Convert a list of dictionaries to a pandas DataFrame.
    Args:
        list_of_dicts (List[dict]): List of dictionaries to convert
        columns_list (List[str]): List of column names to include in DataFrame
    Returns:
        pd.DataFrame: DataFrame containing the unpacked data
    """
    return pd.DataFrame(unpack_lists(list_of_dicts=list_of_dicts, col_list=columns_list))


def plot_img_with_bboxes(img, bboxes) -> None:
    """
    Plot an image_path with overlaid bounding boxes.
    Args:
        img (numpy.ndarray): Image array to plot
        bboxes (List[List[int]]): List of bounding boxes, where each box is [xmin, ymin, xmax, ymax]
    Returns:
        None: Displays the plot with matplotlib
    """

    fig, ax = plt.subplots()
    ax.imshow(img)

    # Add bounding boxes to plot
    for i, bbox in enumerate(bboxes):
        bbox_rectangle = draw_bounding_box(bbox[0], bbox[1], bbox[2], bbox[3],
                                           c.colors_list[i], 2)
        ax.add_patch(bbox_rectangle)

    plt.imshow(img)
    plt.show()

def save_dict_as_json(dict_to_save: dict, output_filepath: str) -> None:
    """Save dictionary to a JSON file at the specified path."""
    with open(output_filepath, 'w') as f:
        json.dump(dict_to_save, f)



def train_test_val_image_split(input_df: pd.DataFrame, test_size: float = 0.2,
                               val_size: float = 0.1) -> dict[str, np.ndarray]:
    """
    Splits the input DataFrame into training, testing, and validation sets
    based on images. If there are multiple entries of the same image_path, it merges them by
    maximum value of each label.

    Parameters:
    ----------
    input_df : pd.DataFrame
        A pandas DataFrame containing column 'img' with file names, bounding box coordinates
        in format (xmin, xmax, ymin, ymax) and their corresponding multi-hot encoded labels.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    val_size : float, default=0.1
        Proportion of the dataset to include in the validation split.

    Returns:
    -------
    dict[str, np.ndarray]
        A dictionary with keys 'train', 'test', and 'val',
        each containing a NumPy array of image_path file names for the respective dataset.
    """
    if input_df.empty:
        raise ValueError('Input DataFrame is empty')

    if not (0 < test_size < 1) or not (0 <= val_size < 1):
        raise ValueError("test_size and val_size must be between 0 and 1.")

    if (test_size + val_size) >= 1:
        raise ValueError("The sum of test_size and val_size must be less than 1.")

    # Create a dictionary to store max values for each image
    grouped_data = {}
    for img_name, group in input_df.groupby(c.IMG):
        grouped_data[img_name] = group[c.defect_names].max().to_numpy()

    # Create new DataFrame with unique images and their labels
    grouped_df = pd.DataFrame(columns=[c.IMG] + c.defect_names)
    grouped_df[c.IMG] = list(grouped_data.keys())
    mhot_labels = np.array(list(grouped_data.values()))

    for i, defect in enumerate(c.defect_names):
        grouped_df[defect] = mhot_labels[:, i]

    # Prepare data for split
    X = grouped_df[c.IMG].to_numpy().reshape(-1, 1)  # Make it 2D array
    y = grouped_df[c.defect_names].to_numpy()

    # Split the data
    X_train, y_train, X_temp, y_temp = iterative_train_test_split(
        X, y, test_size=(1 - test_size + val_size)
    )

    X_test, y_test, X_val, y_val = iterative_train_test_split(
        X_temp, y_temp, test_size=test_size / (test_size + val_size)
    )

    return {
        'train': X_train.flatten(),
        'test': X_test.flatten(),
        'val': X_val.flatten()
    }


def put_imgs_in_folders(train_test_val_dict: dict[str, np.ndarray], input_dir: str, base_out_dir: str) -> None:
    list_of_out_dirs = []
    for k in train_test_val_dict.keys():
        out_dir = os.path.join(base_out_dir, k)
        list_of_out_dirs.append(out_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        elif len(os.listdir(out_dir)) != 0:
            raise ValueError('Output directory {} already exists'.format(out_dir))

        for img_name in train_test_val_dict[k]:
            input_img_path = os.path.join(input_dir, img_name)
            out_img_path = os.path.join(out_dir, img_name)
            shutil.copy(input_img_path, out_img_path)

        print('Copying images to {} finished!'.format(out_dir))


def put_labels_in_folders(train_test_val_dict: dict[str, np.ndarray], input_dir: str, base_out_dir: str) -> None:
    list_of_out_dirs = []
    for k in train_test_val_dict.keys():
        out_dir = os.path.join(base_out_dir, k)
        list_of_out_dirs.append(out_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        elif len(os.listdir(out_dir)) != 0:
            raise ValueError('Output directory {} already exists'.format(out_dir))

        for img_name in train_test_val_dict[k]:
            label_name = img_name.split('.')[0] + '.txt'
            input_img_path = os.path.join(input_dir, label_name)
            out_img_path = os.path.join(out_dir, label_name)
            shutil.copy(input_img_path, out_img_path)

        print('Copying images to {} finished!'.format(out_dir))


def join_cwd(*paths):
    return os.path.join(os.getcwd(), *paths)

def xml_to_dict(filepath: str):
    with open(filepath, 'r') as file:
        xml = file.read()
        return xmltodict.parse(xml)


def copy_imgs(input_dir: str, out_dir: str):
    list_indir = listdir(input_dir)
    for file in tqdm(list_indir):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            copyfile(join(input_dir, file), join(out_dir, file))

