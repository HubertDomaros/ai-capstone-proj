# test_AnnotationsParser
import unittest
from unittest.mock import patch, mock_open
import pandas as pd
from src.preprocessing import AnnotationsParser


class TestAnnotationsParserWithMock(unittest.TestCase):
    def setUp(self):
        # Define the folder path (mocked, so actual value doesn't matter)
        self.folder_path = '/fake/path'

        # Sample XML contents for various test scenarios
        self.sample_xml_multiple_objects = """<?xml version='1.0' encoding='utf-8'?>
        <annotation>
            <folder>images</folder>
            <filename>image_0000001.jpg</filename>
            <size>
                <width>1000</width>
                <height>1000</height>
                <depth>3</depth>
            </size>
            <object>
                <name>defect</name>
                <difficult>0</difficult>
                <bndbox>
                    <xmin>100</xmin>
                    <ymin>200</ymin>
                    <xmax>300</xmax>
                    <ymax>400</ymax>
                </bndbox>
                <Defect>
                    <Background>1</Background>
                    <Crack>0</Crack>
                    <Spallation>0</Spallation>
                    <Efflorescence>0</Efflorescence>
                    <ExposedBars>0</ExposedBars>
                    <CorrosionStain>0</CorrosionStain>
                </Defect>
            </object>
            <object>
                <name>defect</name>
                <difficult>0</difficult>
                <bndbox>
                    <xmin>400</xmin>
                    <ymin>500</ymin>
                    <xmax>600</xmax>
                    <ymax>700</ymax>
                </bndbox>
                <Defect>
                    <Background>0</Background>
                    <Crack>1</Crack>
                    <Spallation>0</Spallation>
                    <Efflorescence>1</Efflorescence>
                    <ExposedBars>0</ExposedBars>
                    <CorrosionStain>0</CorrosionStain>
                </Defect>
            </object>
        </annotation>
        """

        self.sample_xml_single_object = """<?xml version='1.0' encoding='utf-8'?>
        <annotation>
            <folder>images</folder>
            <filename>image_0000002.jpg</filename>
            <size>
                <width>800</width>
                <height>600</height>
                <depth>3</depth>
            </size>
            <object>
                <name>defect</name>
                <difficult>0</difficult>
                <bndbox>
                    <xmin>50</xmin>
                    <ymin>60</ymin>
                    <xmax>200</xmax>
                    <ymax>220</ymax>
                </bndbox>
                <Defect>
                    <Background>0</Background>
                    <Crack>0</Crack>
                    <Spallation>1</Spallation>
                    <Efflorescence>0</Efflorescence>
                    <ExposedBars>1</ExposedBars>
                    <CorrosionStain>0</CorrosionStain>
                </Defect>
            </object>
        </annotation>
        """

        self.sample_xml_no_objects = """<?xml version='1.0' encoding='utf-8'?>
        <annotation>
            <folder>images</folder>
            <filename>image_0000003.jpg</filename>
            <size>
                <width>1024</width>
                <height>768</height>
                <depth>3</depth>
            </size>
        </annotation>
        """

        self.sample_xml_malformed = """<?xml version='1.0' encoding='utf-8'?>
        <annotation>
            <folder>images</folder>
            <filename>image_0000004.jpg</filename>
            <size>
                <width>640</width>
                <height>480</height>
                <depth>3</depth>
            </size>
            <object>
                <name>defect</name>
                <difficult>0</difficult>
                <!-- Missing closing tags intentionally -->
        """

        self.sample_xml_missing_fields = """<?xml version='1.0' encoding='utf-8'?>
        <annotation>
            <folder>images</folder>
            <filename>image_0000005.jpg</filename>
            <size>
                <width>1904</width>
                <height>2856</height>
                <depth>3</depth>
            </size>
            <object>
                <name>defect</name>
                <difficult>0</difficult>
                <bndbox>
                    <xmin>661</xmin>
                    <ymin>472</ymin>
                    <xmax>992</xmax>
                    <ymax>1857</ymax>
                </bndbox>
                <Defect>
                    <!-- Missing some defect fields -->
                    <Efflorescence>1</Efflorescence>
                </Defect>
            </object>
        </annotation>
        """

        self.sample_xml_invalid_defect_values = """<?xml version='1.0' encoding='utf-8'?>
        <annotation>
            <folder>images</folder>
            <filename>image_0000006.jpg</filename>
            <size>
                <width>1904</width>
                <height>2856</height>
                <depth>3</depth>
            </size>
            <object>
                <name>defect</name>
                <difficult>0</difficult>
                <bndbox>
                    <xmin>1507</xmin>
                    <ymin>505</ymin>
                    <xmax>1904</xmax>
                    <ymax>2856</ymax>
                </bndbox>
                <Defect>
                    <Background>invalid</Background>
                    <Crack>0</Crack>
                    <Spallation>0</Spallation>
                    <Efflorescence>1</Efflorescence>
                    <ExposedBars>0</ExposedBars>
                    <CorrosionStain>1</CorrosionStain>
                </Defect>
            </object>
        </annotation>
        """

        self.sample_xml_non_defect_object = """<?xml version='1.0' encoding='utf-8'?>
        <annotation>
            <folder>images</folder>
            <filename>image_0000007.jpg</filename>
            <size>
                <width>500</width>
                <height>500</height>
                <depth>3</depth>
            </size>
            <object>
                <name>non_defect</name>
                <difficult>0</difficult>
                <bndbox>
                    <xmin>10</xmin>
                    <ymin>20</ymin>
                    <xmax>30</xmax>
                    <ymax>40</ymax>
                </bndbox>
                <Defect>
                    <Background>1</Background>
                    <Crack>1</Crack>
                </Defect>
            </object>
        </annotation>
        """

        self.sample_xml_invalid_filename = """<?xml version='1.0' encoding='utf-8'?>
        <annotation>
            <folder>images</folder>
            <filename>invalid_image.jpg</filename>
            <size>
                <width>800</width>
                <height>600</height>
                <depth>3</depth>
            </size>
            <object>
                <name>defect</name>
                <difficult>0</difficult>
                <bndbox>
                    <xmin>100</xmin>
                    <ymin>150</ymin>
                    <xmax>200</xmax>
                    <ymax>250</ymax>
                </bndbox>
                <Defect>
                    <Background>0</Background>
                    <Crack>1</Crack>
                    <Spallation>0</Spallation>
                    <Efflorescence>1</Efflorescence>
                    <ExposedBars>0</ExposedBars>
                    <CorrosionStain>1</CorrosionStain>
                </Defect>
            </object>
        </annotation>
        """

        self.sample_xml_duplicate_filenames = """<?xml version='1.0' encoding='utf-8'?>
        <annotation>
            <folder>images</folder>
            <filename>image_0000008.jpg</filename>
            <size>
                <width>800</width>
                <height>600</height>
                <depth>3</depth>
            </size>
            <object>
                <name>defect</name>
                <difficult>0</difficult>
                <bndbox>
                    <xmin>100</xmin>
                    <ymin>150</ymin>
                    <xmax>200</xmax>
                    <ymax>250</ymax>
                </bndbox>
                <Defect>
                    <Background>0</Background>
                    <Crack>1</Crack>
                    <Spallation>0</Spallation>
                    <Efflorescence>1</Efflorescence>
                    <ExposedBars>0</ExposedBars>
                    <CorrosionStain>1</CorrosionStain>
                </Defect>
            </object>
        </annotation>
        """

    # ----------------------------- Helper Methods ----------------------------- #

    def mock_listdir_factory(file_contents_dict):
        """
        Creates a mock for os.listdir that returns the keys of file_contents_dict as filenames.
        """
        def mock_listdir(path):
            return list(file_contents_dict.keys())
        return mock_listdir

    def mock_open_factory(file_contents_dict):
        """
        Creates a mock_open object that can handle multiple files with different contents.
        """
        def mock_file(file, mode='r', *args, **kwargs):
            filename = os.path.basename(file)
            if filename in file_contents_dict:
                return mock_open(read_data=file_contents_dict[filename]).return_value
            else:
                raise FileNotFoundError(f"No such file: '{file}'")
        return mock_file

    # ----------------------------- Test Cases ----------------------------- #

    @patch('annotations_parser.os.listdir')
    @patch('annotations_parser.open')
    def test_parse_multiple_objects(self, mock_open_func, mock_listdir_func):
        """
        Test parsing an XML file with multiple <object> elements.
        """
        file_contents = {
            'image_0000001.xml': self.sample_xml_multiple_objects
        }
        mock_listdir_func.side_effect = mock_listdir_factory(file_contents)
        mock_open_func.side_effect = mock_open_factory(file_contents)

        parser = AnnotationsParser(self.folder_path)
        df = parser.annotations_df

        # Verify that two entries are created for the two objects
        image_df = df[df['img_name'] == 'image_0000001.jpg']
        self.assertEqual(len(image_df), 2)

        # Validate first object
        first_obj = image_df.iloc[0]
        self.assertEqual(first_obj['xmin'], 100)
        self.assertEqual(first_obj['ymin'], 200)
        self.assertEqual(first_obj['xmax'], 300)
        self.assertEqual(first_obj['ymax'], 400)
        self.assertEqual(first_obj['background'], 1)
        self.assertEqual(first_obj['crack'], 0)
        self.assertEqual(first_obj['efflorescence'], 0)
        self.assertEqual(first_obj['corrosion_stain'], 0)

        # Validate second object
        second_obj = image_df.iloc[1]
        self.assertEqual(second_obj['xmin'], 400)
        self.assertEqual(second_obj['ymin'], 500)
        self.assertEqual(second_obj['xmax'], 600)
        self.assertEqual(second_obj['ymax'], 700)
        self.assertEqual(second_obj['background'], 0)
        self.assertEqual(second_obj['crack'], 1)
        self.assertEqual(second_obj['efflorescence'], 1)
        self.assertEqual(second_obj['corrosion_stain'], 0)

    @patch('annotations_parser.os.listdir')
    @patch('annotations_parser.open')
    def test_parse_single_object(self, mock_open_func, mock_listdir_func):
        """
        Test parsing an XML file with a single <object> element.
        """
        file_contents = {
            'image_0000002.xml': self.sample_xml_single_object
        }
        mock_listdir_func.side_effect = mock_listdir_factory(file_contents)
        mock_open_func.side_effect = mock_open_factory(file_contents)

        parser = AnnotationsParser(self.folder_path)
        df = parser.annotations_df

        # Verify that one entry is created for the single object
        image_df = df[df['img_name'] == 'image_0000002.jpg']
        self.assertEqual(len(image_df), 1)

        # Validate the object
        obj = image_df.iloc[0]
        self.assertEqual(obj['xmin'], 50)
        self.assertEqual(obj['ymin'], 60)
        self.assertEqual(obj['xmax'], 200)
        self.assertEqual(obj['ymax'], 220)
        self.assertEqual(obj['background'], 0)
        self.assertEqual(obj['crack'], 0)
        self.assertEqual(obj['spallation'], 1)
        self.assertEqual(obj['exposed_bars'], 1)
        self.assertEqual(obj['corrosion_stain'], 0)

    @patch('annotations_parser.os.listdir')
    @patch('annotations_parser.open')
    def test_parse_no_objects(self, mock_open_func, mock_listdir_func):
        """
        Test parsing an XML file with no <object> elements.
        """
        file_contents = {
            'image_0000003.xml': self.sample_xml_no_objects
        }
        mock_listdir_func.side_effect = mock_listdir_factory(file_contents)
        mock_open_func.side_effect = mock_open_factory(file_contents)

        parser = AnnotationsParser(self.folder_path)
        df = parser.annotations_df

        # Verify that a background entry is created
        image_df = df[df['img_name'] == 'image_0000003.jpg']
        self.assertEqual(len(image_df), 1)

        # Validate the background entry
        obj = image_df.iloc[0]
        self.assertEqual(obj['background'], 1)
        for col in ['xmin', 'ymin', 'xmax', 'ymax', 'crack',
                    'spallation', 'efflorescence', 'exposed_bars', 'corrosion_stain']:
            self.assertEqual(obj[col], 0)

    @patch('annotations_parser.os.listdir')
    @patch('annotations_parser.open')
    def test_parse_malformed_xml(self, mock_open_func, mock_listdir_func):
        """
        Test parsing a malformed XML file.
        The parser should skip this file, resulting in no entry.
        """
        file_contents = {
            'image_0000004.xml': self.sample_xml_malformed
        }
        mock_listdir_func.side_effect = mock_listdir_factory(file_contents)
        # Simulate a parsing error by raising a ParsingInterrupted exception
        mock_open_func.side_effect = mock_open_factory(file_contents)

        with patch('annotations_parser.xmltodict.parse', side_effect=Exception("Malformed XML")):
            parser = AnnotationsParser(self.folder_path)
            df = parser.annotations_df

            # Verify that no entry is created for the malformed XML
            image_df = df[df['img_name'] == 'image_0000004.jpg']
            self.assertTrue(image_df.empty)

    @patch('annotations_parser.os.listdir')
    @patch('annotations_parser.open')
    def test_parse_missing_fields(self, mock_open_func, mock_listdir_func):
        """
        Test parsing an XML file with missing defect fields.
        Missing fields should default to 0.
        """
        file_contents = {
            'image_0000005.xml': self.sample_xml_missing_fields
        }
        mock_listdir_func.side_effect = mock_listdir_factory(file_contents)
        mock_open_func.side_effect = mock_open_factory(file_contents)

        parser = AnnotationsParser(self.folder_path)
        df = parser.annotations_df

        # Verify that one entry is created
        image_df = df[df['img_name'] == 'image_0000005.jpg']
        self.assertEqual(len(image_df), 1)

        # Validate the entry with missing fields defaulted to 0
        obj = image_df.iloc[0]
        self.assertEqual(obj['xmin'], 661)
        self.assertEqual(obj['ymin'], 472)
        self.assertEqual(obj['xmax'], 992)
        self.assertEqual(obj['ymax'], 1857)
        self.assertEqual(obj['background'], 0)
        self.assertEqual(obj['crack'], 0)
        self.assertEqual(obj['spallation'], 0)
        self.assertEqual(obj['efflorescence'], 1)
        self.assertEqual(obj['exposed_bars'], 0)
        self.assertEqual(obj['corrosion_stain'], 0)  # Missing field defaulted to 0

    @patch('annotations_parser.os.listdir')
    @patch('annotations_parser.open')
    def test_parse_invalid_defect_values(self, mock_open_func, mock_listdir_func):
        """
        Test parsing an XML file with invalid (non-integer) defect values.
        Invalid values should default to 0, and a warning should be logged.
        """
        file_contents = {
            'image_0000006.xml': self.sample_xml_invalid_defect_values
        }
        mock_listdir_func.side_effect = mock_listdir_factory(file_contents)
        mock_open_func.side_effect = mock_open_factory(file_contents)

        with patch('annotations_parser.xmltodict.parse') as mock_parse:
            # Simulate parsing, where 'Background' has an invalid value
            def parse_side_effect(*args, **kwargs):
                return {
                    'annotation': {
                        'filename': 'image_0000006.jpg',
                        'object': {
                            'name': 'defect',
                            'difficult': '0',
                            'bndbox': {
                                'xmin': '1507',
                                'ymin': '505',
                                'xmax': '1904',
                                'ymax': '2856'
                            },
                            'Defect': {
                                'Background': 'invalid',  # Invalid value
                                'Crack': '0',
                                'Spallation': '0',
                                'Efflorescence': '1',
                                'ExposedBars': '0',
                                'CorrosionStain': '1'
                            }
                        }
                    }
                }

            mock_parse.side_effect = parse_side_effect

            with self.assertLogs('annotations_parser', level='WARNING') as cm:
                parser = AnnotationsParser(self.folder_path)
                df = parser.annotations_df

            # Verify that a warning was logged for the invalid Background value
            self.assertIn(
                "WARNING:annotations_parser:Invalid value for Background in image image_0000006.jpg: invalid",
                cm.output
            )

            # Verify that the Background field was defaulted to 0
            image_df = df[df['img_name'] == 'image_0000006.jpg']
            self.assertEqual(len(image_df), 1)
            obj = image_df.iloc[0]
            self.assertEqual(obj['background'], 0)
            self.assertEqual(obj['crack'], 0)
            self.assertEqual(obj['spallation'], 0)
            self.assertEqual(obj['efflorescence'], 1)
            self.assertEqual(obj['exposed_bars'], 0)
            self.assertEqual(obj['corrosion_stain'], 1)

    @patch('annotations_parser.os.listdir')
    @patch('annotations_parser.open')
    def test_fill_missing_images(self, mock_open_func, mock_listdir_func):
        """
        Test that missing images (not present in the XML files) are filled with background data.
        """
        # Assume only image_0000001.xml is present
        file_contents = {
            'image_0000001.xml': self.sample_xml_multiple_objects
        }
        mock_listdir_func.side_effect = mock_listdir_factory(file_contents)
        mock_open_func.side_effect = mock_open_factory(file_contents)

        parser = AnnotationsParser(self.folder_path)
        df = parser.annotations_df

        # Total images expected: 1600
        self.assertEqual(len(df), 1600)

        # Check that background entries are created for missing images
        # For example, image_0000002.jpg should be present as missing and filled with background
        self.assertIn('image_0000002.jpg', df['img_name'].values)
        image_df = df[df['img_name'] == 'image_0000002.jpg']
        self.assertEqual(len(image_df), 1)
        obj = image_df.iloc[0]
        self.assertEqual(obj['background'], 1)
        for col in ['xmin', 'ymin', 'xmax', 'ymax', 'crack',
                    'spallation', 'efflorescence', 'exposed_bars', 'corrosion_stain']:
            self.assertEqual(obj[col], 0)

    @patch('annotations_parser.os.listdir')
    @patch('annotations_parser.open')
    def test_no_xml_files(self, mock_open_func, mock_listdir_func):
        """
        Test the scenario where no XML files are present.
        All images should be treated as missing and filled with background data.
        """
        mock_listdir_func.return_value = []  # No XML files

        parser = AnnotationsParser(self.folder_path)
        df = parser.annotations_df

        # Total images expected: 1600
        self.assertEqual(len(df), 1600)

        # All entries should have background=1 and other fields=0
        self.assertTrue((df['background'] == 1).all())
        for col in ['xmin', 'ymin', 'xmax', 'ymax', 'crack',
                    'spallation', 'efflorescence', 'exposed_bars', 'corrosion_stain']:
            self.assertTrue((df[col] == 0).all())

    @patch('annotations_parser.os.listdir')
    @patch('annotations_parser.open')
    def test_non_defect_objects(self, mock_open_func, mock_listdir_func):
        """
        Test parsing an XML file with a non-defect object.
        Such objects should be ignored, and the image should be treated as missing.
        """
        file_contents = {
            'image_0000007.xml': self.sample_xml_non_defect_object
        }
        mock_listdir_func.side_effect = mock_listdir_factory(file_contents)
        mock_open_func.side_effect = mock_open_factory(file_contents)

        parser = AnnotationsParser(self.folder_path)
        df = parser.annotations_df

        # image_0000007.jpg should be treated as missing and filled with background
        image_df = df[df['img_name'] == 'image_0000007.jpg']
        self.assertEqual(len(image_df), 1)
        obj = image_df.iloc[0]
        self.assertEqual(obj['background'], 1)
        for col in ['xmin', 'ymin', 'xmax', 'ymax', 'crack',
                    'spallation', 'efflorescence', 'exposed_bars', 'corrosion_stain']:
            self.assertEqual(obj[col], 0)

    @patch('annotations_parser.os.listdir')
    @patch('annotations_parser.open')
    def test_invalid_image_filenames(self, mock_open_func, mock_listdir_func):
        """
        Test parsing an XML file with an invalid image filename.
        The parser should still process it correctly.
        """
        file_contents = {
            'invalid_image.xml': self.sample_xml_invalid_filename
        }
        mock_listdir_func.side_effect = mock_listdir_factory(file_contents)
        mock_open_func.side_effect = mock_open_factory(file_contents)

        parser = AnnotationsParser(self.folder_path)
        df = parser.annotations_df

        # 'invalid_image.jpg' should be included in the DataFrame
        image_df = df[df['img_name'] == 'invalid_image.jpg']
        self.assertEqual(len(image_df), 1)

        # Validate the entry
        obj = image_df.iloc[0]
        self.assertEqual(obj['xmin'], 100)
        self.assertEqual(obj['ymin'], 150)
        self.assertEqual(obj['xmax'], 200)
        self.assertEqual(obj['ymax'], 250)
        self.assertEqual(obj['background'], 0)
        self.assertEqual(obj['crack'], 1)
        self.assertEqual(obj['spallation'], 0)
        self.assertEqual(obj['efflorescence'], 1)
        self.assertEqual(obj['exposed_bars'], 0)
        self.assertEqual(obj['corrosion_stain'], 1)

    @patch('annotations_parser.os.listdir')
    @patch('annotations_parser.open')
    def test_duplicate_image_filenames(self, mock_open_func, mock_listdir_func):
        """
        Test parsing multiple XML files with the same image filename.
        Each XML should create a separate entry in the DataFrame.
        """
        file_contents = {
            'image_0000008a.xml': self.sample_xml_duplicate_filenames,
            'image_0000008b.xml': self.sample_xml_duplicate_filenames
        }
        mock_listdir_func.side_effect = mock_listdir_factory(file_contents)
        mock_open_func.side_effect = mock_open_factory(file_contents)

        parser = AnnotationsParser(self.folder_path)
        df = parser.annotations_df

        # 'image_0000008.jpg' should have two entries
        image_df = df[df['img_name'] == 'image_0000008.jpg']
        self.assertEqual(len(image_df), 2)

        # Validate both entries
        for obj in image_df.itertuples(index=False):
            self.assertEqual(obj.xmin, 100)
            self.assertEqual(obj.ymin, 150)
            self.assertEqual(obj.xmax, 200)
            self.assertEqual(obj.ymax, 250)
            self.assertEqual(obj.background, 0)
            self.assertEqual(obj.crack, 1)
            self.assertEqual(obj.spallation, 0)
            self.assertEqual(obj.efflorescence, 1)
            self.assertEqual(obj.exposed_bars, 0)
            self.assertEqual(obj.corrosion_stain, 1)

    @patch('annotations_parser.os.listdir')
    @patch('annotations_parser.open')
    def test_large_number_of_xml_files(self, mock_open_func, mock_listdir_func):
        """
        Test the parser's ability to handle a large number of XML files.
        """
        num_additional_files = 100
        file_contents = {
            f'image_{i:07d}.xml': f"""<?xml version='1.0' encoding='utf-8'?>
            <annotation>
                <folder>images</folder>
                <filename>image_{i:07d}.jpg</filename>
                <size>
                    <width>800</width>
                    <height>600</height>
                    <depth>3</depth>
                </size>
                <object>
                    <name>defect</name>
                    <difficult>0</difficult>
                    <bndbox>
                        <xmin>{i}</xmin>
                        <ymin>{i * 2}</ymin>
                        <xmax>{i + 100}</xmax>
                        <ymax>{i * 2 + 100}</ymax>
                    </bndbox>
                    <Defect>
                        <Background>0</Background>
                        <Crack>1</Crack>
                        <Spallation>0</Spallation>
                        <Efflorescence>1</Efflorescence>
                        <ExposedBars>0</ExposedBars>
                        <CorrosionStain>1</CorrosionStain>
                    </Defect>
                </object>
            </annotation>
            """ for i in range(9, 9 + num_additional_files)
        }
        mock_listdir_func.side_effect = mock_listdir_factory(file_contents)
        mock_open_func.side_effect = mock_open_factory(file_contents)

        parser = AnnotationsParser(self.folder_path)
        df = parser.annotations_df

        # Total images expected: 1600
        self.assertEqual(len(df), 1600)

        # Check a few entries from the additional files
        for i in range(9, 9 + num_additional_files, 10):
            img_name = f'image_{i:07d}.jpg'
            image_df = df[df['img_name'] == img_name]
            self.assertEqual(len(image_df), 1)
            obj = image_df.iloc[0]
            self.assertEqual(obj['xmin'], i)
            self.assertEqual(obj['ymin'], i * 2)
            self.assertEqual(obj['xmax'], i + 100)
            self.assertEqual(obj['ymax'], i * 2 + 100)
            self.assertEqual(obj['background'], 0)
            self.assertEqual(obj['crack'], 1)
            self.assertEqual(obj['spallation'], 0)
            self.assertEqual(obj['efflorescence'], 1)
            self.assertEqual(obj['exposed_bars'], 0)
            self.assertEqual(obj['corrosion_stain'], 1)

    @patch('annotations_parser.os.listdir')
    @patch('annotations_parser.open')
    def test_annotations_df_property(self, mock_open_func, mock_listdir_func):
        """
        Test the annotations_df property to ensure it returns a consistent DataFrame.
        """
        file_contents = {
            'image_0000001.xml': self.sample_xml_multiple_objects,
            'image_0000002.xml': self.sample_xml_single_object
        }
        mock_listdir_func.side_effect = mock_listdir_factory(file_contents)
        mock_open_func.side_effect = mock_open_factory(file_contents)

        parser = AnnotationsParser(self.folder_path)
        df1 = parser.annotations_df
        df2 = parser.annotations_df

        # Ensure that accessing the property multiple times returns the same DataFrame
        pd.testing.assert_frame_equal(df1, df2)

    @patch('annotations_parser.os.listdir')
    @patch('annotations_parser.open')
    def test_non_defect_objects_logging(self, mock_open_func, mock_listdir_func):
        """
        Test that non-defect objects are ignored and background entries are created.
        Additionally, ensure that no warnings are logged for non-defect objects.
        """
        file_contents = {
            'image_0000007.xml': self.sample_xml_non_defect_object
        }
        mock_listdir_func.side_effect = mock_listdir_factory(file_contents)
        mock_open_func.side_effect = mock_open_factory(file_contents)

        with self.assertLogs('annotations_parser', level='WARNING') as cm:
            parser = AnnotationsParser(self.folder_path)
            df = parser.annotations_df

        # Verify that no warnings were logged for non-defect objects
        self.assertEqual(len(cm.output), 0)

        # Verify that the image is treated as missing and filled with background
        image_df = df[df['img_name'] == 'image_0000007.jpg']
        self.assertEqual(len(image_df), 1)
        obj = image_df.iloc[0]
        self.assertEqual(obj['background'], 1)
        for col in ['xmin', 'ymin', 'xmax', 'ymax', 'crack',
                    'spallation', 'efflorescence', 'exposed_bars', 'corrosion_stain']:
            self.assertEqual(obj[col], 0)


if __name__ == '__main__':
    unittest.main()
