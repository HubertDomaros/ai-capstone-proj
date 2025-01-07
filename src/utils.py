import matplotlib.pyplot as plt
from . import constants as c
import pandas as pd
import numpy as np


def draw_bounding_box(xmin:int, ymin:int, xmax:int, ymax:int, edge_color: str = 'blue', linewidth: int=2):
            width = xmax - xmin
            height = ymax - ymin
            x = xmin
            y = ymin
            return plt.Rectangle((x, y), width=width, height=height, 
                                 edgecolor=edge_color, facecolor='none', linewidth=linewidth)
            
            
def unpack_lists(list_of_dicts, col_list) -> dict[str, list[str|int]]:
    out_dict = {}

    for key in col_list:
        out_dict[key] = []

    for single_dict in list_of_dicts:
        for key in single_dict.keys():
            out_dict[key].extend(single_dict[key])

    return out_dict


def list_of_dicts_to_dataframe(list_of_dicts, columns_list) -> pd.DataFrame:
    return pd.DataFrame(unpack_lists(list_of_dicts=list_of_dicts, col_list=columns_list))


def plot_img_with_bboxes(img, bboxes) -> None:
    """
    Plot the processed image with bounding boxes.
    """
    fig, ax = plt.subplots()
    ax.imshow(img)
    for bbox in bboxes:
        i=0
        bbox_rectangle = draw_bounding_box(bbox[0], bbox[1], bbox[2], bbox[3],
                                                    c.colors_list[i], 2)
        ax.add_patch(bbox_rectangle)
        i+=1
    plt.imshow(img)
    plt.show()