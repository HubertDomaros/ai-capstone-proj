from dataclasses import dataclass
from bs4 import BeautifulSoup
from bs4.element import Tag

@dataclass
class XmlDefect:
    """
    A data class representing a defect parsed from XML annotations.
    
    This class should only be instantiated using the from_tag() class method, 
    not initialized directly.
    
    Attributes:
        bounding_box (tuple[int, int, int, int]): Coordinates of defect bounding box 
            in format (xmin, xmax, ymin, ymax)
        crack (bool): Whether defect contains a crack
        spallation (bool): Whether defect contains spallation damage
        efflorescence (bool): Whether defect contains efflorescence (salt deposits)
        exposed_bars (bool): Whether defect contains exposed reinforcement bars
        corrosion_stain (bool): Whether defect contains corrosion staining
    
    Example:
        >>> defect = XmlDefect.from_tag(object_tag)
    """
    bounding_box: tuple[int, int, int, int]
    crack: bool
    spallation: bool
    efflorescence: bool
    exposed_bars: bool
    corrosion_stain: bool
    
    @classmethod
    def from_tag(cls, object_tag: Tag) -> 'XmlDefect':
        """
        Creates an XmlDefect instance from an XML object tag.

        Args:
            object_tag (bs4.element.Tag): XML tag <object> containing defect annotation data.
                The tag must contain bndbox, Crack, Spallation, Efflorescence, ExposedBars,
                and CorrosionStain child elements.

        Returns:
            XmlDefect: A new instance containing the parsed defect data.

        Raises:
            ValueError: If provided tag is not an 'object' tag.
        """
        if object_tag.name != "object":
            raise ValueError(f"Tag passed to the method is not <object> tag. Received: <{object_tag.name}>")
        
        xmin: int = int(object_tag.bndbox.xmin.text)
        xmax: int = int(object_tag.bndbox.xmax.text)
        ymin: int = int(object_tag.bndbox.ymin.text)
        ymax: int = int(object_tag.bndbox.ymax.text)
        
        return cls(
            bounding_box = (xmin, xmax, ymin, ymax),
            crack=bool(int(object_tag.Crack.text)),
            spallation=bool(int(object_tag.Spallation.text)),
            efflorescence=bool(int(object_tag.Efflorescence.text)),
            exposed_bars=bool(int(object_tag.ExposedBars.text)),
            corrosion_stain = bool(int(object_tag.CorrosionStain.text))
        )
    
    
@dataclass    
class XmlParser:
    pass