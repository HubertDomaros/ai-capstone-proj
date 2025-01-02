import os
import re
import pandas as pd
import xmltodict
import logging
from xml.parsers.expat import ExpatError

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class AnnotationsParser:
    def __init__(self, folder_path: str):
        self._folder_path = folder_path
        self._cols = [
            'img_name', 'xmin', 'ymin', 'xmax', 'ymax',
            'Background', 'Crack', 'Spallation', 'Efflorescence',
            'ExposedBars', 'CorrosionStain'
        ]
        self._annotations_df = None

    @staticmethod
    def parse_xml_to_dict(filepath: str) -> dict | None:
        try:
            with open(filepath, 'r') as file:
                return xmltodict.parse(file.read())
        except (FileNotFoundError, xmltodict.ParsingInterrupted, ExpatError) as e:
            logger.warning(f"Error parsing file {filepath}: {e}")
            return None

    def parse_dict_annotation(self, json_dict: dict) -> pd.DataFrame:
        img_name = json_dict['annotation'].get('filename', '')
        out_dict = {col: [] for col in self._cols}
        objects = json_dict['annotation'].get('object', [])

        if isinstance(objects, dict):
            objects = [objects]

        defects_found = False

        # Parse each object from the XML
        for obj in objects:
            if obj.get('name') == 'defect':
                defects_found = True
                out_dict['img_name'].append(img_name)

                # Bounding box - fill with 0 if missing
                bndbox = obj.get('bndbox', {})
                for coord in ('xmin', 'ymin', 'xmax', 'ymax'):
                    out_dict[coord].append(int(bndbox.get(coord, 0)))

                # Parse defects and handle missing values
                defects = obj.get('Defect', {})
                for defect in self._cols:
                    if defect not in ('img_name', 'xmin', 'ymin', 'xmax', 'ymax'):
                        try:
                            out_dict[defect].append(int(defects.get(defect, 0)))
                        except ValueError:
                            logger.warning(f"Invalid value for {defect} in {img_name}: {defects.get(defect)}")
                            out_dict[defect].append(0)

        # If no defects found, add background row
        if not defects_found:
            bg_row = self.create_background_dict(img_name)
            for key, val in bg_row.items():
                out_dict[key].append(val[0])

        return pd.DataFrame(out_dict)

    def create_background_dict(self, img_name: str) -> dict:
        row = {col: [0] for col in self._cols}
        row['img_name'] = [img_name]
        row['Background'] = [1]  # Default to background=1
        return row

    def fill_df_with_missing_images(self, df: pd.DataFrame) -> pd.DataFrame:
        existing_imgs = set(df['img_name'])
        last_img_name = df['img_name'].tolist()[-1][6:13]
        last_img_no_re = re.search(r'(\d+)', last_img_name)
        last_img_no = int(last_img_no_re.group(1)) if last_img_no_re else 0
        all_imgs = {f"image{img:07d}.jpg" for img in range(1, last_img_no)}
        missing_imgs = all_imgs - existing_imgs

        background_dicts = [self.create_background_dict(img) for img in missing_imgs]
        background_df = pd.DataFrame(background_dicts)

        return pd.concat([df, background_df], ignore_index=True)

    def parse_xmls_to_dataframe(self) -> pd.DataFrame:
        df_list = []
        files = os.listdir(self._folder_path)

        for file in files:
            if file.endswith('.xml'):
                filepath = os.path.join(self._folder_path, file)
                xml = self.parse_xml_to_dict(filepath)
                if xml:
                    df = self.parse_dict_annotation(xml)
                    if not df.empty:
                        df_list.append(df)

        return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame(columns=self._cols)

    def initialize_annotations(self):
        parsed_folder = self.parse_xmls_to_dataframe()
        filled_df = self.fill_df_with_missing_images(parsed_folder)
        self._annotations_df = filled_df

    @property
    def annotations_df(self) -> pd.DataFrame:
        if self._annotations_df is None:
            self.initialize_annotations()
        return self._annotations_df
