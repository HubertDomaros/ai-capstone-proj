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

    img_name = []
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    Background = []
    Crack = []
    Spallation = []
    Efflorescence = []
    ExposedBars = []
    CorrosionStain = []

    if 'object' not in annotation:
        img_name.append(annotation['filename'])
        xmin.append(0)
        ymin.append(0)
        xmax.append(int(annotation['size']['width']))
        ymax.append(int(annotation['size']['height']))
        Background.append(1)
        Crack.append(0)
        Spallation.append(0)
        Efflorescence.append(0)
        ExposedBars.append(0)
        CorrosionStain.append(0)

        out_dict = {
            'img': img_name,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'Background': Background,
            'Crack': Crack,
            'Spallation': Spallation,
            'Efflorescence': Efflorescence,
            'ExposedBars': ExposedBars,
            'CorrosionStain': CorrosionStain
        }

        return out_dict

    bbox_descriptions = []

    if type(annotation['object']) is list:
        bbox_descriptions = annotation['object']
    else:
        bbox_descriptions = [annotation['object']]

    for bbox_description in bbox_descriptions:
        img_name.append(annotation['filename'])
        xmin.append(int(bbox_description['bndbox']['xmin']))
        ymin.append(int(bbox_description['bndbox']['ymin']))
        xmax.append(int(bbox_description['bndbox']['xmax']))
        ymax.append(int(bbox_description['bndbox']['ymax']))

        Background.append(int(bbox_description['Defect']['Background']))
        Crack.append(bbox_description['Defect']['Crack'])
        Spallation.append(int(bbox_description['Defect']['Spallation']))
        Efflorescence.append(int(bbox_description['Defect']['Efflorescence']))
        ExposedBars.append(int(bbox_description['Defect']['ExposedBars']))
        CorrosionStain.append(int(bbox_description['Defect']['CorrosionStain']))

        out_dict = {
            'img': img_name,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'Background': Background,
            'Crack': Crack,
            'Spallation': Spallation,
            'Efflorescence': Efflorescence,
            'ExposedBars': ExposedBars,
            'CorrosionStain': CorrosionStain
        }

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


print(x.head())