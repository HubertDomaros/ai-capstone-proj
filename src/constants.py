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
class StdColNames(Enum):
    IMG = IMG
    WIDTH = WIDTH
    HEIGHT = HEIGHT
    XMIN = XMIN
    YMIN = YMIN
    XMAX = XMAX
    YMAX = YMAX
    BACKGROUND = BACKGROUND
    CRACK = CRACK
    SPALLATION = SPALLATION
    EFFLORESCENCE = EFFLORESCENCE
    EXPOSEDBARS = EXPOSEDBARS
    CORROSIONSTAIN = CORROSIONSTAIN

colums_list = [IMG, WIDTH, HEIGHT, XMIN, YMIN, YMIN, YMAX,
               BACKGROUND, SPALLATION, EFFLORESCENCE, EXPOSEDBARS, CORROSIONSTAIN]

class Dimensions(Enum):
    WIDTH = WIDTH
    HEIGHT = HEIGHT
    XMIN = XMIN
    YMIN = YMIN
    XMAX = XMAX
    YMAX = YMAX


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

class DefectNames(Enum):
    BACKGROUND = BACKGROUND
    CRACK = CRACK
    SPALLATION = SPALLATION
    EFFLORESCENCE = EFFLORESCENCE
    EXPOSEDBARS = EXPOSEDBARS
    CORROSIONSTAIN = CORROSIONSTAIN