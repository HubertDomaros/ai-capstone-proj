import numpy as np
import os
from preprocessing import annotation_extractor as an_ext
from preprocessing import constants as c

def convert_bboxes_to_albumentations(
    bboxes: np.ndarray,
    source_format: str = "pascal_voc",
    shape: tuple[int, int] = (0, 0),
    check_validity: bool = False,
) -> np.ndarray:

    bboxes = bboxes.copy().astype(np.float32)
    converted_bboxes = np.zeros_like(bboxes)
    converted_bboxes[:, 4:] = bboxes[:, 4:]  # Preserve additional columns
    
    converted_bboxes[:, :4] = bboxes[:, :4]

    if source_format != "yolo":
        converted_bboxes[:, :4] = normalize_bboxes(converted_bboxes[:, :4], shape)

    return converted_bboxes


def normalize_bboxes(bboxes: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Normalize array of bounding boxes.

    Args:
        bboxes: Denormalized bounding boxes `[(x_min, y_min, x_max, y_max, ...)]`.
        shape: Image shape `(height, width)`.

    Returns:
        Normalized bounding boxes `[(x_min, y_min, x_max, y_max, ...)]`.

    """
    if isinstance(shape, tuple):
        rows, cols = shape[:2]
    else:
        rows, cols = shape["height"], shape["width"]

    normalized = bboxes.copy().astype(float)
    normalized[:, [0, 2]] /= cols
    normalized[:, [1, 3]] /= rows
    return normalized




dataset_path = r'/kaggle/input/codebrim-original/original_dataset'
annotations_path = os.path.join(dataset_path, r'annotations')
adf = an_ext.xml_annotations_to_dataframe(annotations_path)
adf[['xmin','ymin', 'xmax', 'ymax']].astype(int)


img_arr = adf['img'].apply(lambda x: f'/kaggle/input/codebrim-original/original_dataset/images/{x}').to_numpy()
img_shapes_arr = adf[['height', 'width']].to_numpy()
bboxes_pascal_arr = adf[c.bbox_coordinate_names].to_numpy()
labels_arr = adf[c.defect_names].to_numpy()

print(bboxes_pascal_arr)
print(img_arr)