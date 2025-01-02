import os
import re
import pandas as pd
import xmltodict
import logging
from xml.parsers.expat import ExpatError
from typing import Optional, List, Dict

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class AnnotationsParser:
    def __init__(self, folder_path: str) -> None:
        self._folder_path: str = folder_path
        self._cols: List[str] = [
            'img_name', 'xmin', 'ymin', 'xmax', 'ymax',
            'Background', 'Crack', 'Spallation', 'Efflorescence',
            'ExposedBars', 'CorrosionStain'
        ]
        self._annotations_df: Optional[pd.DataFrame] = None

    @staticmethod
    def parse_xml_to_dict(filepath: str) -> Optional[Dict[str, any]]:
        try:
            with open(filepath, 'r') as file:
                return xmltodict.parse(file.read())
        except (FileNotFoundError, xmltodict.ParsingInterrupted, ExpatError) as e:
            logger.warning(f"Error parsing file {filepath}: {e}")
            return None

    def parse_dict_annotation(self, json_dict: Dict[str, any]) -> pd.DataFrame:
        img_name: str = json_dict['annotation'].get('filename', '')
        out_dict: Dict[str, List[any]] = {col: [] for col in self._cols}
        objects: List[Dict[str, any]] = json_dict['annotation'].get('object', [])

        if isinstance(objects, dict):
            objects = [objects]

        defects_found: bool = False

        # Parse each object from the XML
        for obj in objects:
            if obj.get('name') == 'defect':
                defects_found = True
                out_dict['img_name'].append(img_name)

                # Bounding box - fill with 0 if missing
                bndbox: Dict[str, str] = obj.get('bndbox', {})
                for coord in ('xmin', 'ymin', 'xmax', 'ymax'):
                    out_dict[coord].append(int(bndbox.get(coord, 0)))

                # Parse defects and handle missing values
                defects: Dict[str, str] = obj.get('Defect', {})
                for defect in self._cols:
                    if defect not in ('img_name', 'xmin', 'ymin', 'xmax', 'ymax'):
                        try:
                            out_dict[defect].append(int(defects.get(defect, 0)))
                        except ValueError:
                            logger.warning(f"Invalid value for {defect} in {img_name}: {defects.get(defect)}")
                            out_dict[defect].append(0)

        # If no defects found, add background row
        if not defects_found:
            bg_row: Dict[str, str | int] = self.create_background_dict(img_name)
            for key, val in bg_row.items():
                out_dict[key].append(val)

        return pd.DataFrame(out_dict)

    def create_background_dict(self, img_name: str) -> Dict[str, str | int]:
        row: Dict[str, str | int] = {col: 0 for col in self._cols}
        row['img_name'] = img_name
        row['Background'] = 1  # Default to background=1
        return row

    def fill_df_with_missing_images(self, df: pd.DataFrame) -> pd.DataFrame:
        existing_imgs: set[str] = set(df['img_name'])
        last_img_name: str = df['img_name'].tolist()[-1][6:13]
        last_img_no_re: Optional[re.Match] = re.search(r'(\d+)', last_img_name)
        last_img_no: int = int(last_img_no_re.group(1)) if last_img_no_re else 0
        all_imgs: set[str] = {f"image{img:07d}.jpg" for img in range(1, last_img_no)}
        missing_imgs: set[str] = all_imgs - existing_imgs

        background_dicts: List[Dict[str, str | int]] = [self.create_background_dict(img) for img in missing_imgs]
        background_df: pd.DataFrame = pd.DataFrame(background_dicts)

        return pd.concat([df, background_df], ignore_index=True)

    def parse_xmls_to_dataframe(self) -> pd.DataFrame:
        df_list: List[pd.DataFrame] = []
        files: List[str] = os.listdir(self._folder_path)

        for file in files:
            if file.endswith('.xml'):
                filepath: str = os.path.join(self._folder_path, file)
                xml: Optional[Dict[str, any]] = self.parse_xml_to_dict(filepath)
                if xml:
                    df: pd.DataFrame = self.parse_dict_annotation(xml)
                    if not df.empty:
                        df_list.append(df)

        return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame(columns=self._cols)

    def initialize_annotations(self) -> None:
        parsed_folder: pd.DataFrame = self.parse_xmls_to_dataframe()
        filled_df: pd.DataFrame = self.fill_df_with_missing_images(parsed_folder)
        self._annotations_df = filled_df

    @property
    def annotations_df(self) -> pd.DataFrame:
        if self._annotations_df is None:
            self.initialize_annotations()
        return self._annotations_df
