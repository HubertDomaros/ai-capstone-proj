from enum import Enum


IMG = 'img'
WIDTH = 'width'
HEIGHT = 'height'
XMIN = 'xmin'
YMIN = 'ymin'
XMAX = 'xmax'
YMAX = 'ymax'
BACKGROUND = 'Background'
CRACK = 'Crack'
SPALLATION = 'Spallation'
EFFLORESCENCE = 'Efflorescence'
EXPOSEDBARS = 'ExposedBars'
CORROSIONSTAIN = 'CorrosionStain'

columns_list = [IMG, WIDTH, HEIGHT, XMIN, YMIN, XMAX, YMAX,
                BACKGROUND, CRACK, SPALLATION, EFFLORESCENCE, EXPOSEDBARS, CORROSIONSTAIN]

image_dims_names = [WIDTH, HEIGHT]
bbox_coordinate_names = [XMIN, YMIN, XMAX, YMAX]
defect_names = [BACKGROUND, CRACK, SPALLATION, EFFLORESCENCE, EXPOSEDBARS, CORROSIONSTAIN]

class Colors(Enum):
    BLUE = 'tab:blue'
    ORANGE = 'tab:orange'
    GREEN = 'tab:green'
    RED = 'tab:red'
    PURPLE = 'tab:purple'
    BROWN = 'tab:brown'
    PINK = 'tab:pink'
    GRAY = 'tab:gray'
    OLIVE = 'tab:olive'
    CYAN = 'tab:cyan'

colors_list = [Colors.BLUE.value, Colors.ORANGE.value, Colors.GREEN.value, Colors.RED.value, Colors.PURPLE.value,
            Colors.BROWN.value, Colors.PINK.value, Colors.GRAY.value, Colors.OLIVE.value, Colors.CYAN.value]