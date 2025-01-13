import os
import re

import xmltodict
import imagesize
import pandas as pd
import numpy as np
from skmultilearn.model_selection import iterative_train_test_split

from . import constants as c
from . import utils as u


# bbox coordinates: object -> bndbox -> xmin, xmax, ymin, ymax
# defect types: object -> Defect -> Background, Crack, Spallation, Efflorescence, ExposedBars, CorrosionStain
def xml_to_dict(filepath: str):
    with open(filepath, 'r') as file:
        xml = file.read()
        return xmltodict.parse(xml)


def parse_bounding_boxes_labels(dict_with_labels: dict) -> dict[str, list[str | int]]:
    annotation = dict_with_labels['annotation']
    img_name = annotation['filename']
    size = annotation['size']
    size[c.IMG_WIDTH] = size.pop('width')
    size[c.IMG_HEIGHT] = size.pop('height')

    objects = annotation.get('object', [])

    if not isinstance(objects, list):
        objects = [objects]

    out_dict = {col: [] for col in c.pascal_cols_list}

    # Default values for images with no objects
    if not objects:

        out_dict[c.IMG] = [img_name]
        out_dict[c.IMG_WIDTH] = [int(size[c.IMG_WIDTH])]
        out_dict[c.IMG_HEIGHT] = [int(size[c.IMG_HEIGHT])]
        out_dict[c.BACKGROUND] = [1]

        for key in c.bbox_coordinate_names:
            out_dict[key] = [0]

        for key in c.defect_names[1:]:
            out_dict[key] = [0]

    for obj in objects:
        bbox = obj['bndbox']
        defect = obj['Defect']

        out_dict[c.IMG].append(img_name)

        for key in c.image_dims_names:
            out_dict[key].append(int(size[key]))

        for key in c.bbox_coordinate_names:
            out_dict[key].append(int(bbox[key]))

        for key in c.defect_names:
            out_dict[key].append(int(defect[key]))

    return out_dict


def extract_annotations_from_xmls(folder_path: str) -> list[dict[str, list[str | int]]]:
    list_of_files = os.listdir(folder_path)
    list_of_dicts_of_bbox_descriptions = []

    for file_path in list_of_files:
        if file_path.endswith('.xml'):
            converted_xml = xml_to_dict(filepath=os.path.join(folder_path, file_path))
            bbox_descriptions = parse_bounding_boxes_labels(dict_with_labels=converted_xml)
            list_of_dicts_of_bbox_descriptions.append(bbox_descriptions)

    return list_of_dicts_of_bbox_descriptions


def xml_annotations_to_dataframe(folder_path: str) -> pd.DataFrame:
    list_of_dicts = extract_annotations_from_xmls(folder_path=folder_path)
    lst = u.unpack_lists(list_of_dicts=list_of_dicts, col_list=c.pascal_cols_list)
    return pd.DataFrame(lst)


def fill_missing_imgs_in_df(img_folder_path, input_df):
    """
    Adds missing images from a folder to a DataFrame containing image_path metadata.
    For each image_path in the folder that's not in the input DataFrame, gets its dimensions
    and creates a new row with default values. Returns concatenated DataFrame with new entries.

    Args:
        img_folder_path (str): Path to folder containing images
        input_df (pd.DataFrame): DataFrame with existing image_path metadata

    Returns:
        pd.DataFrame: Original DataFrame with new rows for missing images,
        or original DataFrame if no new images found
    """
    # Get list of images
    img_list = []
    for dirpath, dirnames, filenames in os.walk(img_folder_path):
        if dirpath == img_folder_path:
            img_list = filenames
            break

    # Process images sequentially
    results = []
    input_imgs_from_df = input_df[c.IMG].tolist()
    for img in img_list:

        if img not in input_df[c.IMG].tolist() and img.split('.')[-1] == 'img':
            img_path = os.path.join(os.getcwd(), img_folder_path, img)
            # using awesome imagesize lib! Super fast! super cool! Supageil!
            # https://github.com/shibukawa/imagesize_py
            shape = imagesize.get(img_path)
            l = [0] * len(c.pascal_cols_list)
            l[0] = img
            l[1] = shape[0]
            l[2] = shape[1]
            l[7] = 1
            results.append(l)

    # Filter out None results and add to DataFrame
    out_df = 0
    out_df = pd.concat([input_df, pd.DataFrame(results, columns=c.pascal_cols_list)], ignore_index=True)
    return out_df


def generate_multihot_encoding_combinations(n) -> list[tuple[int, list[int]]]:
    """Generate all possible multi-hot encoded combinations for n classes.
    Args:
        n (int): Number of classes
    Returns:
        list: List of lists containing all possible multi-hot combinations
    """
    combinations: list[tuple[int, list[int]]] = []

    # Generate 2^n combinations (0 to 2^n-1)
    for i in range(2 ** n):
        # Convert number to binary and remove '0b' prefix
        binary = format(i, f'0{n}b')
        # Convert binary string to list of integers
        multihot = [int(b) for b in binary]
        combinations.append((i, multihot))

    return combinations


def convert_single_pascal_bbox_to_yolo(bbox, img_height, img_width, multihot_label):
    xmin = bbox[0] / img_width
    ymin = bbox[1] / img_height
    xmax = bbox[2] / img_width
    ymax = bbox[3] / img_height

    bbox_width = xmax - xmin
    bbox_height = ymax - ymin

    x_center = (xmax + xmin) / 2
    y_center = (ymax + ymin) / 2

    label_class = c.possible_multihot_encodings.index(multihot_label)

    return {
        c.MULTIHOT_ENCODING_CLASS: label_class,
        c.BBOX_X_CENTER: x_center,
        c.BBOX_Y_CENTER: y_center,
        c.BBOX_WIDTH: bbox_width,
        c.BBOX_HEIGHT: bbox_height
    }


def pascal_df_to_yolo(df: pd.DataFrame) -> pd.DataFrame:
    out_df = df.copy().drop(c.bbox_coordinate_names, axis=1)

    converted_bboxes = {
        c.MULTIHOT_ENCODING_CLASS: [],
        c.BBOX_X_CENTER: [],
        c.BBOX_Y_CENTER: [],
        c.BBOX_WIDTH: [],
        c.BBOX_HEIGHT: []
    }
    for iterrow in df.iterrows():
        row = iterrow[1]
        width = row[c.IMG_WIDTH]
        height = row[c.IMG_HEIGHT]
        bbox = row[c.bbox_coordinate_names].tolist()

        multihot_encoding = row[c.defect_names].tolist()
        converted_bbox = (convert_single_pascal_bbox_to_yolo(bbox, width, height,
                                                             multihot_encoding))

        for key, value in converted_bbox.items():
            converted_bboxes[key].append(value)

    for key, value in converted_bboxes.items():
        out_df[key] = value

    return out_df


def save_yolo_annotations(df: pd.DataFrame, out_folder: str):
    i = 0

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    elif len(os.listdir(out_folder)) != 0:
        raise OSError("Output folder is not empty")

    for img, data in df.groupby('img'):
        img = str(img)
        filename = f'{img.split(".")[0]}.txt'
        output = data[[c.MULTIHOT_ENCODING_CLASS,
                       c.BBOX_X_CENTER, c.BBOX_Y_CENTER, c.BBOX_WIDTH, c.BBOX_HEIGHT]]
        output.to_csv(os.path.join(out_folder, filename), index=False, header=False, sep=" ")
        i += 1

    print(f'Saved {i} annotations to {out_folder}')



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

    # Group by image and combine labels
    grouped_df = input_df.groupby(c.IMG).agg({
        c.defect_names: lambda x: tuple(np.array(x.tolist()).max(axis=0).tolist())
    }).reset_index()

    print(grouped_df)

    input_df_cols = input_df.columns.tolist()

    # is_original_df = all(defect in input_df_cols for defect in c.pascal_cols_list)
    mhot_encoding = grouped_df[c.defect_names].to_numpy()

    y = mhot_encoding
    X = grouped_df[c.IMG].to_numpy()

    X_train, y_train, X_temp, y_temp = iterative_train_test_split(
        X, y, test_size=(1 - test_size + val_size)
    )

    X_test, y_test, X_val, y_val = iterative_train_test_split(
        X_temp, y_temp, test_size=test_size / (test_size + val_size)
    )

    return {
        'train': X_train,
        'test': X_test,
        'val': X_val
    }




