import os
import cv2
import matplotlib.pyplot as plt
import albumentations as A
import numpy.typing as npt
import pandas as pd
import numpy as np
import albumentations.core.bbox_utils as bu
from src import annotation_extractor as an_ext
from src import constants as c
import src.image_processing as improc
import src.utils as u




def augment_image_and_save_to_folder(input_file_path: str, output_folder: str, data: pd.DataFrame) -> dict:
    bboxes = data[c.bbox_coordinate_names].astype(int).to_numpy()
    multi_hot_encoded_labels = [list(x) for x in data[c.defect_names].astype(int).to_numpy()]
    # bboxes = bu.convert_bboxes_to_albumentations(bboxes, 'pascal_voc', shape=(2856, 1904))



    augmented_images: list[improc.ImageAugumentor] = improc.generate_augmented_images(image_path=input_file_path,
                                                                                      bounding_boxes=bboxes,
                                                                                      label_values=multi_hot_encoded_labels,
                                                                                      resize=True, out_width=512,
                                                                                      out_height=512)

    out_dict_list = []
    for augmented_image in augmented_images:
        cv2.imwrite(os.path.join(output_folder, augmented_image.processed_image_name), augmented_image.processed_image)
        out_dict_list.append(augmented_image.metadata_dict)

    return u.unpack_lists(out_dict_list, out_dict_list[0].keys())


in_fpath = os.path.join(os.getcwd(), 'examples', 'image_0000005.jpg')
out_f = r'D:\0-Code\PG\2_sem\0_Dyplom\ai-capstone-proj\examples\augumented'

df: pd.DataFrame = an_ext.xml_annotations_to_dataframe('examples')
datas = 0
for fname, data in df.groupby('img')[list(c.columns_list[1:])]:
    datas = data
a = augment_image_and_save_to_folder(in_fpath, out_f, datas)

print(a)
