from enum import Enum

IMG = 'img'
IMG_WIDTH = 'img_width'
IMG_HEIGHT = 'img_height'
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

pascal_cols_list = [IMG, IMG_WIDTH, IMG_HEIGHT,
                    XMIN, YMIN, XMAX, YMAX,
                    BACKGROUND, CRACK, SPALLATION, EFFLORESCENCE, EXPOSEDBARS, CORROSIONSTAIN]

image_dims_names = [IMG_WIDTH, IMG_HEIGHT]
bbox_coordinate_names = [XMIN, YMIN, XMAX, YMAX]
defect_names = [BACKGROUND, CRACK, SPALLATION, EFFLORESCENCE, EXPOSEDBARS, CORROSIONSTAIN]

BBOX_X_CENTER = 'x_center'
BBOX_Y_CENTER = 'y_center'
BBOX_HEIGHT = 'bbox_height'
BBOX_WIDTH = 'bbox_width'
MULTIHOT_ENCODING_CLASS = 'multihot_encoding_class'

yolo_cols_list = [IMG, IMG_WIDTH, IMG_HEIGHT,
                  BBOX_X_CENTER, BBOX_Y_CENTER, BBOX_WIDTH, BBOX_HEIGHT, MULTIHOT_ENCODING_CLASS,
                  BACKGROUND, CRACK, SPALLATION, EFFLORESCENCE, EXPOSEDBARS, CORROSIONSTAIN]

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

possible_multihot_encodings = (
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 1],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 1],
    [0, 0, 0, 1, 1, 0],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 1],
    [0, 0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1, 1],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 1],
    [0, 0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1],
    [0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 1],
    [0, 1, 0, 0, 1, 0],
    [0, 1, 0, 0, 1, 1],
    [0, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 1],
    [0, 1, 0, 1, 1, 0],
    [0, 1, 0, 1, 1, 1],
    [0, 1, 1, 0, 0, 0],
    [0, 1, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0],
    [0, 1, 1, 0, 1, 1],
    [0, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0, 1],
    [0, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 1, 1],
    [1, 0, 0, 1, 0, 0],
    [1, 0, 0, 1, 0, 1],
    [1, 0, 0, 1, 1, 0],
    [1, 0, 0, 1, 1, 1],
    [1, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1, 1],
    [1, 0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 0],
    [1, 0, 1, 1, 1, 1],
    [1, 1, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 1],
    [1, 1, 0, 0, 1, 0],
    [1, 1, 0, 0, 1, 1],
    [1, 1, 0, 1, 0, 0],
    [1, 1, 0, 1, 0, 1],
    [1, 1, 0, 1, 1, 0],
    [1, 1, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 1],
    [1, 1, 1, 0, 1, 0],
    [1, 1, 1, 0, 1, 1],
    [1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1]
)