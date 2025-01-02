from numpy import sin, cos, radians
import pandas as pd
import numpy as np
from numpy.typing import NDArray
import cv2

def transform_bounding_box(
        img_width: int, img_height: int,
        xmin: int, xmax: int, ymin: int, ymax: int, angle: float, angle_in_radians: bool
) -> NDArray[int]:
    """
    Rotate each corner of a rectangular bounding box around its center by the specified angle.

    The corners of the bounding box are assumed to be:
      (xmin, ymin)
        (x1, y1) -------- (x4, y4)
            |                |
            |                |
        (x2, y2) -------- (x3, y3)
                        (xmax, ymax)

    Parameters:
        img_width (int): width of the image
        img_height (int): height of the image
        xmin (int): The minimum X coordinate of the bounding box.
        xmax (int): The maximum X coordinate of the bounding box.
        ymin (int): The minimum Y coordinate of the bounding box.
        ymax (int): The maximum Y coordinate of the bounding box.
        angle (float): The rotation angle (in degrees unless otherwise specified in transform_coordinates).
        angle_in_radians (bool): Determines whether the rotation angle is in radians or degrees.

    Returns:
        NDArray[int]:
            The rotated coordinates of the bounding box in the order:
            (x1_prim, y1_prim, x2_prim, y2_prim, x3_prim, y3_prim, x4_prim, y4_prim).
    """
    x1, y1 = xmin, ymin
    x2, y2 = xmin, ymax
    x3, y3 = xmax, ymax
    x4, y4 = xmax, ymin

    if angle == 0:
        return np.array([x1, y1, x2, y2, x3, y3, x4, y4], dtype=int)

    c1: NDArray[int] = transform_coordinates(x1, y1, angle, angle_in_radians)
    c2: NDArray[int] = transform_coordinates(x2, y2, angle, angle_in_radians)
    c3: NDArray[int] = transform_coordinates(x3, y3, angle, angle_in_radians)
    c4: NDArray[int] = transform_coordinates(x4, y4, angle, angle_in_radians)

    return np.append(c1, [c2, c3, c4]).astype(int)


def transform_coordinates(
        x: int, y: int, img_width: int, img_height: int,
        angle: float, angle_in_radians: bool = False) -> NDArray[int]:
    """
    Rotate a point (x, y) around the origin (0, 0) by a given angle.

    Parameters:
        x (int): The original X coordinate of the point.
        y (int): The original Y coordinate of the point.
        angle (float): The angle for rotation.
        angle_in_radians (bool): Whether `angle` is already in radians. If False,
            `angle` is treated as degrees and automatically converted to radians.

    Returns:
        NDArray[int]:
            The new coordinates (x_prim, y_prim) of the point after rotation,
            rounded down (floored) to integers.
    """

    x_center = img_width / 2
    y_center = img_height / 2

    x_dist_from_center = x_center - x
    y_dist_from_center = y_center - y

    if not angle_in_radians:
        angle = radians(angle)

    x_prim: int = int(round(x_dist_from_center * cos(angle) + y_dist_from_center * sin(angle)))
    y_prim: int = int(round(x_dist_from_center * sin(angle) + y_dist_from_center * cos(angle)))

    x_out = x_prim + x_center
    y_out = y_prim + y_center

    if x_out < 0:
        raise ValueError(f'x coordinate cannot be negative! current value: x={x_out}')
    if y_out < 0:
        raise ValueError(f'y coordinate cannot be negative! current value: y={y_out}')

    return np.array([x_out, y_out], dtype=int)


def change_df_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts bounding box coordinates from (xmin, xmax, ymin, ymax) to four-point format
    (x1, y1, x2, y2, x3, y3, x4, y4) and appends them to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing bounding box coordinates with columns
         ['xmin', 'xmax', 'ymin', 'ymax'].

    Returns:
        pd.DataFrame: DataFrame with new columns representing transformed bounding box
        coordinates, preserving other data present in the original DataFrame.
    """
    bbox_list: list[NDArray[int]] = []
    old_bbox: list[str] = ['xmin', 'xmax', 'ymin', 'ymax']
    new_bbox: list[str] = ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']

    old_cols: list[str] = df.drop(columns=old_bbox).columns.tolist()
    new_cols: list[str] = old_cols + new_bbox

    for index, row in df.iterrows():
        bbox: NDArray[int] = transform_bounding_box(
            row['xmin'], row['xmax'], row['ymin'], row['ymax'], 0.0, False
        )
        bbox_list.append(bbox)

    if not bbox_list:
        x = [0] * 8
        bbox_list: NDArray[int] = np.empty((0, 8), dtype=int)

    new_df: pd.DataFrame = pd.DataFrame(columns=new_cols)
    new_df[new_bbox] = pd.DataFrame(bbox_list, index=df.index)

    return new_df

def create_mask_from_bboxes(image_shape, bboxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    mask = np.zeros(image_shape[:2], dtype=np.uint8)  # Initialize mask (HxW)
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = map(int, bbox)
        cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 1, thickness=-1)  # Fill box with 1
    return mask
