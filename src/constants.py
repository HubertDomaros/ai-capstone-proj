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
BBOX = 'bbox'

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
    [(0, [0, 0, 0, 0, 0, 0]),
     (1, [0, 0, 0, 0, 0, 1]),
     (2, [0, 0, 0, 0, 1, 0]),
     (3, [0, 0, 0, 0, 1, 1]),
     (4, [0, 0, 0, 1, 0, 0]),
     (5, [0, 0, 0, 1, 0, 1]),
     (6, [0, 0, 0, 1, 1, 0]),
     (7, [0, 0, 0, 1, 1, 1]),
     (8, [0, 0, 1, 0, 0, 0]),
     (9, [0, 0, 1, 0, 0, 1]),
     (10, [0, 0, 1, 0, 1, 0]),
     (11, [0, 0, 1, 0, 1, 1]),
     (12, [0, 0, 1, 1, 0, 0]),
     (13, [0, 0, 1, 1, 0, 1]),
     (14, [0, 0, 1, 1, 1, 0]),
     (15, [0, 0, 1, 1, 1, 1]),
     (16, [0, 1, 0, 0, 0, 0]),
     (17, [0, 1, 0, 0, 0, 1]),
     (18, [0, 1, 0, 0, 1, 0]),
     (19, [0, 1, 0, 0, 1, 1]),
     (20, [0, 1, 0, 1, 0, 0]),
     (21, [0, 1, 0, 1, 0, 1]),
     (22, [0, 1, 0, 1, 1, 0]),
     (23, [0, 1, 0, 1, 1, 1]),
     (24, [0, 1, 1, 0, 0, 0]),
     (25, [0, 1, 1, 0, 0, 1]),
     (26, [0, 1, 1, 0, 1, 0]),
     (27, [0, 1, 1, 0, 1, 1]),
     (28, [0, 1, 1, 1, 0, 0]),
     (29, [0, 1, 1, 1, 0, 1]),
     (30, [0, 1, 1, 1, 1, 0]),
     (31, [0, 1, 1, 1, 1, 1]),
     (32, [1, 0, 0, 0, 0, 0])]
)