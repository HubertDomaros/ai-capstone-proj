import os

import cv2
import pandas as pd

from src import constants as c
import src.original.image_processing as improc
import src.utils as u

def augment_image_and_save_to_folder(input_file_path: str, output_folder: str, data: pd.DataFrame) -> dict:
    bboxes = data[c.bbox_coordinate_names].astype(int).to_numpy()
    multi_hot_encoded_labels = [list(x) for x in data[c.defect_names].astype(int).to_numpy()]

    # Generate augmented image_path variations
    augmented_images = improc.generate_augmented_images(
        image_path=input_file_path,
        bounding_boxes=bboxes,
        label_values=multi_hot_encoded_labels,
        resize=True,
        target_width=512,
        target_height=512
    )

    # Save each augmented image_path and collect metadata
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




