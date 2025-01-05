import matplotlib.pyplot as plt
import itertools
from collections.abc import Iterable


def draw_bounding_box(xmin:int, ymin:int, xmax:int, ymax:int, edge_color: str = 'blue', linewidth: int=2):
            width = xmax - xmin
            height = ymax - ymin
            x = xmin
            y = ymin
            return plt.Rectangle((x, y), width=width, height=height, 
                                 edgecolor=edge_color, facecolor='none', linewidth=linewidth)
            
            
def unpack_iterable(iterable: Iterable):
    return list(itertools.chain.from_iterable(iterable))

def unpack_lists(list_of_dicts) -> dict[str, list[str, int]]:
    out_dict = []
    
    for single_dict in list_of_dicts:
        if not isinstance(single_dict, dict):
            print(f"Skipping invalid entry: {single_dict}")
            continue
        
        try:
            for key in single_dict.keys():
                out_dict[key] = []
                for value in single_dict[key]:
                    out_dict[key].append(value)
        except Exception as e:
            print('Exception occurred on dict', single_dict)
            raise Exception(e)

    return out_dict