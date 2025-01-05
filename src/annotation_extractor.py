import os

import xmltodict
import pandas as pd

import constants as c

# bbox coordinates: object -> bndbox -> xmin, xmax, ymin, ymax
# defect types: object -> Defect -> Background, Crack, Spallation, Efflorescence, ExposedBars, CorrosionStain
def xml_to_dict(filepath: str):
    with open(filepath, 'r') as file:
        xml = file.read()
        return xmltodict.parse(xml)


def parse_bounding_boxes_labels_inside_image(dict_with_labels: dict) -> dict[str, list[str|int]]:
    annotation = dict_with_labels['annotation']
    img_name = annotation['filename']
    size = annotation['size']

    objects = annotation.get('object', [])

    if not isinstance(objects, list):
        objects = [objects]

    # Initialize output dictionary
    # out_dict = {
    #         'img': [],
    #         'width': [],
    #         'height': [],
    #         'xmin': [],
    #         'ymin': [],
    #         'xmax': [],
    #         'ymax': [],
    #         'Background': [],
    #         'Crack': [],
    #         'Spallation': [],
    #         'Efflorescence': [],
    #         'ExposedBars': [],
    #         'CorrosionStain': []
    #     }

    out_dict = {}

    for key in c.StdColNames:
        out_dict[key] = []

    # Default values for images with no objects
    if not objects:

        out_dict = {
            c.IMG: [img_name],
            c.WIDTH: [int(size['width'])],
            c.HEIGHT: [int(size['height'])],
            c.XMIN: [0],
            c.YMIN: [0],
            c.XMAX: [int(size['width'])],
            c.YMAX: [int(size['height'])],
            c.BACKGROUND: [1]
        }

        for key in c.DefectNames[1:]:
            out_dict[key] = [0]


    for obj in objects:
        bbox = obj['bndbox']
        defect = obj['Defect']

        out_dict[c.IMG].append(img_name)

        for key in c.StdColNames[1:]:
            out_dict[key].append(int(defect[key]))

    return out_dict


def extract_annotations_from_xmls(folder_path: str) -> list[dict[str, list[str | int]]]:
    list_of_files = os.listdir(folder_path)
    list_of_dicts_of_bbox_descriptions = []

    for file_path in list_of_files:
        if file_path.endswith('.xml'):
            converted_xml = xml_to_dict(filepath=os.path.join(folder_path, file_path))
            bbox_descriptions = parse_bounding_boxes_labels_inside_image(dict_with_labels=converted_xml)
            list_of_dicts_of_bbox_descriptions.append(bbox_descriptions)

    return list_of_dicts_of_bbox_descriptions


def unpack_lists(list_of_dicts) -> dict[str, list[str, int]]:
    out_dict = {
        'img': [],
        'width': [],
        'height': [],
        "xmin": [],
        "ymin": [],
        "xmax": [],
        "ymax": [],
        "Background": [],
        "Crack": [],
        "Spallation": [],
        "Efflorescence": [],
        "ExposedBars": [],
        "CorrosionStain": []
    }

    for single_dict in list_of_dicts:
        if not isinstance(single_dict, dict):
            print(f"Skipping invalid entry: {single_dict}")
            continue
        try:
            for key in single_dict.keys():
                for value in single_dict[key]:
                    out_dict[key].append(value)
        except Exception as e:
            print('Exception occurred on dict', single_dict)
            raise Exception(e)

    return out_dict


def xml_annotations_to_dataframe(folder_path: str) -> pd.DataFrame:
    list_of_dicts = extract_annotations_from_xmls(folder_path=folder_path)
    lst = unpack_lists(list_of_dicts=list_of_dicts)
    return pd.DataFrame(lst)


if __name__ == '__main__':
    x = xml_annotations_to_dataframe(
        r'D:\0-Code\PG\2_sem\0_Dyplom\ai-capstone-proj\kaggle\input\codebrim-original\original_dataset\annotations')
    print(x)
