import os

import xmltodict
import pandas as pd

# bbox coordinates: object -> bndbox -> xmin, xmax, ymin, ymax
# defect types: object -> Defect -> Background, Crack, Spallation, Efflorescence, ExposedBars, CorrosionStain

example_dict = {
        'img': 'image_0000005.jpg',
        "xmin": 661,
        "ymin": 472,
        "xmax": 992,
        "ymax": 1857,
        "Background": 0,
        "Crack": 0,
        "Spallation": 0,
        "Efflorescence": 1,
        "ExposedBars": 0,
        "CorrosionStain": 1
    }

def xml_to_dict(filepath: str):
    with open(filepath, 'r') as file:
        xml = file.read()
        return xmltodict.parse(xml)


def parse_bounding_boxes_labels_inside_image(dict_with_labels: dict):
    annotation = dict_with_labels['annotation']
    img_name = annotation['filename']
    size = annotation['size']
    objects = annotation.get('object', [])

    if not isinstance(objects, list):
        objects = [objects]

    # Default values for images with no objects
    if not objects:
        return {
            'img': [img_name],
            'xmin': [0], 'ymin': [0],
            'xmax': [int(size['width'])],
            'ymax': [int(size['height'])],
            'Background': [1],
            'Crack': [0],
            'Spallation': [0],
            'Efflorescence': [0],
            'ExposedBars': [0],
            'CorrosionStain': [0]
        }

    # Initialize output dictionary
    out_dict = {
        'img': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
        'Background': [], 'Crack': [], 'Spallation': [],
        'Efflorescence': [], 'ExposedBars': [], 'CorrosionStain': []
    }

    for obj in objects:
        bbox = obj['bndbox']
        defect = obj['Defect']

        out_dict['img'].append(img_name)
        out_dict['xmin'].append(int(bbox['xmin']))
        out_dict['ymin'].append(int(bbox['ymin']))
        out_dict['xmax'].append(int(bbox['xmax']))
        out_dict['ymax'].append(int(bbox['ymax']))

        out_dict['Background'].append(int(defect['Background']))
        out_dict['Crack'].append(int(defect['Crack']))
        out_dict['Spallation'].append(int(defect['Spallation']))
        out_dict['Efflorescence'].append(int(defect['Efflorescence']))
        out_dict['ExposedBars'].append(int(defect['ExposedBars']))
        out_dict['CorrosionStain'].append(int(defect['CorrosionStain']))

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
    out_dict =  {
        'img': [],
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


x = xml_annotations_to_dataframe(r'D:\0-Code\PG\2_sem\0_Dyplom\ai-capstone-proj\kaggle\input\codebrim-original\original_dataset\annotations')


print(x)