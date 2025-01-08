import matplotlib.pyplot as plt
from . import constants as c
import pandas as pd
import numpy as np
import json


def draw_bounding_box(xmin: int, ymin: int, xmax: int, ymax: int, edge_color: str = 'blue', linewidth: int = 2):
    """
    Create a rectangular bounding box for visualization.
    Args:
        xmin (int): Minimum x-coordinate of the bounding box
        ymin (int): Minimum y-coordinate of the bounding box
        xmax (int): Maximum x-coordinate of the bounding box
        ymax (int): Maximum y-coordinate of the bounding box
        edge_color (str, optional): Color of the bounding box edge. Defaults to 'blue'
        linewidth (int, optional): Width of the bounding box line. Defaults to 2
    Returns:
        plt.Rectangle: A matplotlib Rectangle patch representing the bounding box
    """
    width = xmax - xmin
    height = ymax - ymin
    x = xmin
    y = ymin
    return plt.Rectangle((x, y), width=width, height=height,
                        edgecolor=edge_color, facecolor='none', linewidth=linewidth)


def unpack_lists(list_of_dicts, col_list) -> dict[str, list[str | int]]:
    """
    Unpack a list of dictionaries into a single dictionary with lists as values.
    Handles numpy arrays by converting them to lists.
    Args:
        list_of_dicts (List[dict]): List of dictionaries to unpack
        col_list (List[str]): List of column names to include in output
    Returns:
        dict[str, list[str|int]]: Dictionary with column names as keys and lists of values
    """

    out_dict = {}
    for key in col_list:
        out_dict[key] = []

    # Process each dictionary in the input list
    for single_dict in list_of_dicts:
        for key in single_dict.keys():
            val = single_dict[key]
            if isinstance(val, np.ndarray):
                val = val.tolist()
            out_dict[key].extend(val)

    return out_dict


def list_of_dicts_to_dataframe(list_of_dicts, columns_list) -> pd.DataFrame:
    """
    Convert a list of dictionaries to a pandas DataFrame.
    Args:
        list_of_dicts (List[dict]): List of dictionaries to convert
        columns_list (List[str]): List of column names to include in DataFrame
    Returns:
        pd.DataFrame: DataFrame containing the unpacked data
    """
    return pd.DataFrame(unpack_lists(list_of_dicts=list_of_dicts, col_list=columns_list))


def plot_img_with_bboxes(img, bboxes) -> None:
    """
    Plot an image with overlaid bounding boxes.
    Args:
        img (numpy.ndarray): Image array to plot
        bboxes (List[List[int]]): List of bounding boxes, where each box is [xmin, ymin, xmax, ymax]
    Returns:
        None: Displays the plot with matplotlib
    """

    fig, ax = plt.subplots()
    ax.imshow(img)

    # Add bounding boxes to plot
    for i, bbox in enumerate(bboxes):
        bbox_rectangle = draw_bounding_box(bbox[0], bbox[1], bbox[2], bbox[3],
                                           c.colors_list[i], 2)
        ax.add_patch(bbox_rectangle)

    plt.imshow(img)
    plt.show()

def save_dict_as_json(dict_to_save: dict, output_filepath: str) -> None:
    """Save dictionary to a JSON file at the specified path."""
    with open(output_filepath, 'w') as f:
        json.dump(dict_to_save, f)