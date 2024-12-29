from dataclasses import dataclass
from bs4 import BeautifulSoup
from bs4.element import Tag

@dataclass
class XmlDefect():
    """
    A data class representing a defect parsed from XML annotations.
    
    Attributes:
        bounding_box (tuple[int, int, int, int]): Coordinates of defect bounding box (xmin, xmax, ymin, ymax)
        crack (bool): Whether defect contains a crack
        spallation (bool): Whether defect contains spallation
        efflorescence (bool): Whether defect contains efflorescence  
        exposed_bars (bool): Whether defect contains exposed reinforcement bars
        corrosion_stain (bool): Whether defect contains corrosion staining
    
    Args:
        object_tag (bs4.element.Tag): XML tag <object> containing defect annotation data
        
    Raises:
        ValueError: If provided tag is not an 'object' tag
    """
    bounding_box: tuple[int, int, int, int]
    crack: bool
    spallation: bool
    efflorescence: bool
    exposed_bars: bool
    corrosion_stain: bool
    
    def __post_init__(self, object_tag: Tag):
        if object_tag.text is not "object":
            raise ValueError(f"Tag passed to the method is not <object> tag.
                              Recieved: <{object_tag.text}>")
        
        xmin: int = int(object_tag.bndbox.xmin.text)
        xmax: int = int(object_tag.bndbox.xmax.text)
        ymin: int = int(object_tag.bndbox.ymin.text)
        ymax: int = int(object_tag.bndbox.ymax.text)
        self.bounding_box = (xmin, xmax, ymin, ymax)
        
        self.crack=object_tag.crack.text,
        self.spallation=object_tag.spallation.text,
        self.efflorescence=object_tag.efflorescence.text,
        self.exposed_bars=object_tag.exposed_bars.text,
        self.joint_seperation=object_tag.joint_seperation.text
        
