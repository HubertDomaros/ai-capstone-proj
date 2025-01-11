import os
import re

import xmltodict
import pandas as pd

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

# def fill_missing_imgs_as_backgrounds(df: pd.DataFrame) -> pd.DataFrame:
#     img_list
#
#     first = int(re.search('\d+', img_list[0]).group(0))
#     last = int(re.search('\d+', img_list[-1]).group(0))






if __name__ == '__main__':
    x = xml_annotations_to_dataframe(
        r'D:\0-Code\PG\2_sem\0_Dyplom\ai-capstone-proj\kaggle\input\codebrim-original\original_dataset\annotations')
    print(x)
