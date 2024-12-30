import os
from bs4 import BeautifulSoup
from bs4.element import Tag
import pandas as pd

class XmlDefect:
    """
    A class representing a defect parsed from XML annotations.

    This class should be initialized with an XML `object_tag`.

    Attributes:
        bounding_box (tuple[int, int, int, int]): Coordinates of defect bounding box 
            in format (xmin, xmax, ymin, ymax)
        crack (bool): Whether defect contains a crack
        spallation (bool): Whether defect contains spallation damage
        efflorescence (bool): Whether defect contains efflorescence (salt deposits)
        exposed_bars (bool): Whether defect contains exposed reinforcement bars
        corrosion_stain (bool): Whether defect contains corrosion staining
    """

    def __init__(self, object_tag, image_name):
        """
        Initializes an XmlDefect instance from an XML <object> tag.

        Args:
            object_tag (bs4.element.Tag): XML tag <object> containing defect annotation data.
            The tag must contain bndbox, Crack, Spallation, Efflorescence, ExposedBars,
            and CorrosionStain child elements.

        Raises:
            ValueError: If provided tag is not an 'object' tag.
        """
        if object_tag.name != "object":
            raise ValueError(f"Tag passed to the constructor is not <object> tag. Received: <{object_tag.name}>")
        
        self._image_name = object_tag.
        self._bounding_box = (
            int(object_tag.bndbox.xmin.text),
            int(object_tag.bndbox.xmax.text),
            int(object_tag.bndbox.ymin.text),
            int(object_tag.bndbox.ymax.text)
        )
        self._crack = bool(int(object_tag.Crack.text))
        self._spallation = bool(int(object_tag.Spallation.text))
        self._efflorescence = bool(int(object_tag.Efflorescence.text))
        self._exposed_bars = bool(int(object_tag.ExposedBars.text))
        self._corrosion_stain = bool(int(object_tag.CorrosionStain.text))

    @property
    def bounding_box(self):
        return self._bounding_box

    @property
    def crack(self):
        return self._crack

    @property
    def spallation(self):
        return self._spallation

    @property
    def efflorescence(self):
        return self._efflorescence

    @property
    def exposed_bars(self):
        return self._exposed_bars

    @property
    def corrosion_stain(self):
        return self._corrosion_stain
    
    def to_series(self) -> pd.Series:
        dc = {
            'xmin' : self._bounding_box[0],
            'xmax' : self._bounding_box[1],
            'ymin' : self._bounding_box[2],
            'ymax' : self._bounding_box[3],
            'crack': self._crack,
            'spallation': self._spallation,
            'efflorescence': self._efflorescence,
            'exposed_bars': self._exposed_bars,
            'corrosion_stain': self._corrosion_stain
        }
        return pd.Series(dc)


def defect_to_series(image_name: str, object_tag:Tag) -> pd.Series:
    object_dict = {
    'image': image_name,
    "xmin": int(object_tag.bndbox.xmin.text),
    "xmax": int(object_tag.bndbox.xmax.text),
    "ymin": int(object_tag.bndbox.ymin.text),
    "ymax": int(object_tag.bndbox.ymax.text),
    "crack": int(object_tag.Crack.text),
    "spallation": int(object_tag.Spallation.text),
    "efflorescence": int(object_tag.Efflorescence.text),
    "exposed_bars": int(object_tag.ExposedBars.text),
    "corrosion_stain": int(object_tag.CorrosionStain.text)
}


class XmlDefects:
    def __init__(self, filepath):
        self._filepath = filepath
        self._image_filename = ""
        self._defects: list[XmlDefect] = []
        self._file_content: Tag = None
        self._parse()

    def _parse(self):
        with open(self._filepath, 'r') as file:
            xml = BeautifulSoup(file, 'xml')
            if not xml.is_xml:
                raise ValueError(f"File {self._filepath} is not a valid XML file.")

            self._file_content = xml.find('annotation')  # Note: changed from get to find
            self._image_filename = xml.find('filename').text

            for defect in xml.find_all('object'):
                self._defects.append(XmlDefect.from_tag(defect))

    @property
    def file_content(self):
        return self._file_content

    @property
    def img_filename(self):
        return self._image_filename

    @property
    def defects(self):
        return self._defects

def parse_folder_with_xmls(folder_path: str):
    files = os.listdir(folder_path)
    parsed_folder: list[XmlDefects] = []

    df = pd.DataFrame()
    pd.columns

    
    for file in files:
        if file.endswith('.xml'):
            filepath = os.path.join(folder_path, file)
            xml_defects = XmlDefects(filepath)
            parsed_folder.append(xml_defects)
    return parsed_folder



