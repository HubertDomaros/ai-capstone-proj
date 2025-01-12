import os
import re

import xmltodict
import pandas as pd
import imagesize

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

    objects = annotation.get('object', [])

    if not isinstance(objects, list):
        objects = [objects]

    out_dict = {col:[] for col in c.columns_list}

    # Default values for images with no objects
    if not objects:

        out_dict[c.IMG] = [img_name]
        out_dict[c.WIDTH] = [int(size['width'])]
        out_dict[c.HEIGHT] = [int(size['height'])]
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
            out_dict[key].append(size[key])

        for key in c.bbox_coordinate_names:
            out_dict[key].append(bbox[key])

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
    lst = u.unpack_lists(list_of_dicts=list_of_dicts, col_list=c.columns_list)
    return pd.DataFrame(lst)


def fill_missing_imgs_in_df(img_folder_path, input_df):
    """
    Adds missing images from a folder to a DataFrame containing image metadata.
    For each image in the folder that's not in the input DataFrame, gets its dimensions
    and creates a new row with default values. Returns concatenated DataFrame with new entries.

    Args:
        img_folder_path (str): Path to folder containing images
        input_df (pd.DataFrame): DataFrame with existing image metadata

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
    for img in img_list:
        if img not in input_df['img'].tolist():
            img_path = os.path.join(os.getcwd(), img_folder_path, img)
            # using awesome imagesize lib! Super fast! super cool!
            # https://github.com/shibukawa/imagesize_py
            shape = imagesize.get(img_path)
            l = [0] * len(c.columns_list)
            l[0] = img
            l[1] = shape[0]
            l[2] = shape[1]
            l[7] = 1
            results.append(l)

    # Filter out None results and add to DataFrame
    out_df = 0
    out_df = pd.concat([input_df, pd.DataFrame(results, columns=c.columns_list)], ignore_index=True)
    return out_df

def save_yolo_annotations(df: pd.DataFrame, out_folder: str):
    i = 0
    for img, data in df.groupby('img'):
        img = str(img)
        filename = f'{img.split('.')[0]}.txt'
        output = data[['multi_hot_encoding_class',
                                'xcenter', 'ycenter', 'width', 'height']]
        output.to_csv(os.path.join(out_folder, filename), index=False, header=False, sep = " ")
        i += 1

    print(f'Saved {i} annotations to {out_folder}')


if __name__ == '__main__':
    x = xml_annotations_to_dataframe(
        r'D:\0-Code\PG\2_sem\0_Dyplom\ai-capstone-proj\kaggle\input\codebrim-original\original_dataset\annotations')
    print(x)
