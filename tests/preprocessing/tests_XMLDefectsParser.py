import unittest
import os
import pandas as pd
from unittest.mock import patch, mock_open
from src.preprocessing.XMLDefectsParser import XMLDefectsParser  # Replace 'your_module' with the actual module name

class TestXMLDefectsParser(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = 'test_xml_folder'
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        # Remove the temporary directory after tests
        for file in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, file))
        os.rmdir(self.test_dir)

    def test_initialization(self):
        parser = XMLDefectsParser(self.test_dir)
        self.assertEqual(len(parser.defect_df), 0)
        self.assertEqual(len(parser.background_filepaths), 0)

    def test_parse_valid_xml(self):
        xml_content = """<annotation>
                            <filename>image1.jpg</filename>
                            <object>
                                <bndbox>
                                    <xmin>10</xmin>
                                    <xmax>20</xmax>
                                    <ymin>30</ymin>
                                    <ymax>40</ymax>
                                </bndbox>
                                <Crack>1</Crack>
                                <Spallation>0</Spallation>
                                <Efflorescence>0</Efflorescence>
                                <ExposedBars>0</ExposedBars>
                                <CorrosionStain>0</CorrosionStain>
                            </object>
                        </annotation>"""
        with open(os.path.join(self.test_dir, 'test1.xml'), 'w') as f:
            f.write(xml_content)

        parser = XMLDefectsParser(self.test_dir)
        self.assertEqual(len(parser.defect_df), 1)
        self.assertEqual(parser.defect_df.iloc[0]['image'], 'image1.jpg')
        self.assertEqual(parser.defect_df.iloc[0]['crack'], 1)

    def test_parse_background_xml(self):
        xml_content = """<annotation>
                            <filename>image2.jpg</filename>
                        </annotation>"""
        with open(os.path.join(self.test_dir, 'test2.xml'), 'w') as f:
            f.write(xml_content)

        parser = XMLDefectsParser(self.test_dir)
        self.assertEqual(len(parser.background_filepaths), 1)
        self.assertIn(os.path.join(self.test_dir, 'test2.xml'), parser.background_filepaths)

    def test_parse_invalid_xml(self):
        with open(os.path.join(self.test_dir, 'invalid.xml'), 'w') as f:
            f.write("<invalid></invalid>")

        parser = XMLDefectsParser(self.test_dir)
        self.assertEqual(len(parser.defect_df), 0)
        self.assertEqual(len(parser.background_filepaths), 0)

    def test_error_handling(self):
        with patch('builtins.open', side_effect=FileNotFoundError):
            parser = XMLDefectsParser(self.test_dir)
            self.assertEqual(len(parser.defect_df), 0)
            self.assertEqual(len(parser.background_filepaths), 0)

if __name__ == '__main__':
    unittest.main()
