import os
import pandas as pd
import xmltodict
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class AnnotationsParser:
    def __init__(self, folder_path: str):
        self._folder_path = folder_path
        self._cols = ['img_name', 'xmin', 'ymin', 'xmax', 'ymax',
                        'background', 'crack', 'spallation', 'efflorescence', 
                        'exposed_bars', 'corrosion_stain']
        self._annotations_df = self._initialize_annotations()

    @staticmethod
    def parse_xml_to_dict(filepath: str) -> dict|None:
        try:
            with open(filepath, 'r') as file:
                return xmltodict.parse(file.read())
        except (FileNotFoundError, xmltodict.ParsingInterrupted) as e:
            print(f"Error parsing file {filepath}: {e}")
            return None
        
    def parse_dict_annotation(self, json_dict: dict) -> pd.DataFrame:
        img_name = json_dict['annotation']['filename']
        out_dict = {col: [] for col in self._cols}
        objects = json_dict['annotation'].get('object', [])
        
        defects_found = False
        
        for obj in objects:
            if obj['name'] != 'defect':
                continue
            
            defects_found = True

            out_dict['img_name'].append(img_name)
            
            # Parse bounding box coordinates
            bndbox = obj.get('bndbox', {})
            for coord, value in bndbox.items():
                out_dict[coord].append(int(value))

            # Parse defects
            defects: dict[str] = obj.get('Defect', {})
            for defect, value in defects.items():
                try:
                    out_dict[defect].append(int(value))
                except ValueError:
                    logger.warning(f"Invalid value for {defect} in image {img_name}: {value}")
                    out_dict[defect].append(0)

        if not defects_found:
            return pd.DataFrame([self.create_background_dict(img_name)])

        return pd.DataFrame(out_dict)

    def create_background_dict(self, img_name: str) -> dict:
        out_dict = {col: [] for col in self._cols}
        out_dict['img_name'].append(img_name)
        out_dict['background'].append(1)

        for key in self._cols:
            if key not in ['img_name', 'background']:
                out_dict[key].append(0)

        return out_dict
    
    def fill_df_with_missing_images(self, df: pd.DataFrame) -> pd.DataFrame:
        existing_imgs = set(df['img_name'])
        all_imgs = {f"image{img:07d}" for img in range(1, 1601)}
        missing_imgs = all_imgs - existing_imgs

        background_dicts = [self.create_background_dict(img_name) for img_name in missing_imgs]
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
        if df_list:
            return pd.concat(df_list, ignore_index=True)
        else:
            return pd.DataFrame(columns=self._cols)

    def _initialize_annotations(self) -> pd.DataFrame:
        parsed_folder = self.parse_xmls_to_dataframe()
        filled_df = self.fill_df_with_missing_images(parsed_folder)
        return filled_df
    
    @property
    def annotations_df(self) -> pd.DataFrame:
        return self._annotations_df
