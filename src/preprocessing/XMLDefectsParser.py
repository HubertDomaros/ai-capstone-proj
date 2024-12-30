import os
from bs4 import BeautifulSoup
from bs4.element import Tag
import pandas as pd

class XMLDefectsParser:
    def __init__(self, folder_path: str):
        self._folder_path = folder_path
        self._defect_list: list[dict] = []
        self._background_list: list[str] = []
        self._defect_df: pd.DataFrame = pd.DataFrame()  # Initialize an empty DataFrame
        self._parse_folder()

    def _defect_xml_to_dict(self, image_name: str, object_tag: Tag) -> dict:
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
        return object_dict

    def _parse_xml_with_defects(self, filepath: str) -> None:
        try:
            with open(filepath, 'r') as file:
                xml = BeautifulSoup(file, 'xml')
                if len(xml.find_all('annotation')) == 0:
                    self._background_list.append(filepath)
                    return

                image_name: str = xml.find('filename').text

                for defect in xml.find_all('object'):
                    self._defect_list.append(self._defect_xml_to_dict(image_name, defect))
        except Exception as e:
            print(f"Error processing file {filepath}: {e}")

    def _parse_folder(self) -> None:
        try:
            files = os.listdir(self._folder_path)
            for file in files:
                if file.endswith('.xml'):
                    filepath = os.path.join(self._folder_path, file)
                    self._parse_xml_with_defects(filepath)

            # Create the DataFrame after parsing all files
            self._defect_df = pd.DataFrame(self._defect_list, 
                                            columns=['image', 'xmin', 'xmax', 'ymin', 'ymax', 
                                                     'crack', 'spallation', 'efflorescence', 
                                                     'exposed_bars', 'corrosion_stain'])
        except Exception as e:
            print(f"Error reading folder {self._folder_path}: {e}")

    @property
    def defect_df(self) -> pd.DataFrame:
        return self._defect_df

    @property
    def background_filepaths(self) -> list[str]:
        return self._background_list

# Example usage:
# parser = XMLDefectParser('path/to/xml/folder')
# parser.parse_folder()
# defects_df = parser.defect_df
# background_files = parser.background_filepaths
