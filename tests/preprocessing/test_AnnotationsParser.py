import os
import pytest
import pandas as pd
from unittest.mock import patch, mock_open
from src.preprocessing.AnnotationsParser import AnnotationsParser



################################################################################
#                          HELPER FACTORIES FOR MOCKING                        #
################################################################################

def mock_listdir_factory(file_contents_dict):
    """
    Utility factory that creates a mock listdir function.
    file_contents_dict is a dict like:
      {
        'file1.xml': '<annotation> ... </annotation>',
        'file2.xml': '<annotation> ... </annotation>',
      }
    """

    def mock_listdir(path):
        return list(file_contents_dict.keys())

    return mock_listdir


def mock_open_factory(file_contents_dict):
    """
    Utility factory that creates a mock_open function which returns
    the file contents from `file_contents_dict` for a given filename.
    """

    def _open(file, mode='r', *args, **kwargs):
        filename = os.path.basename(file)
        data = file_contents_dict.get(filename, "")
        return mock_open(read_data=data)()

    return _open


################################################################################
#                        ORIGINAL BASIC TESTS (from your code)                 #
################################################################################

def test_parse_xml_to_dict_valid():
    """
    Test that parse_xml_to_dict returns a valid dictionary if the file exists and is well-formed.
    """
    fake_xml_content = """<?xml version='1.0' encoding='utf-8'?>
        <annotation>
            <folder>images</folder>
            <filename>image_0000005.jpg</filename>
        </annotation>
    """
    with patch("builtins.open", mock_open(read_data=fake_xml_content)) as mocked_file:
        parser = AnnotationsParser(folder_path="fake_folder")
        data = parser.parse_xml_to_dict("fake_file.xml")
        assert data is not None
        assert data['annotation']['folder'] == 'images'
        assert data['annotation']['filename'] == 'image_0000005.jpg'


def test_parse_xml_to_dict_file_not_found():
    """
    Test that parse_xml_to_dict returns None if the file is not found.
    """
    with patch("builtins.open", side_effect=FileNotFoundError("File not found")):
        parser = AnnotationsParser(folder_path="fake_folder")
        data = parser.parse_xml_to_dict("non_existent.xml")
        assert data is None


@pytest.fixture
def sample_xml_dict():
    """
    Returns a dictionary (similar to the output of xmltodict.parse)
    corresponding to the sample XML provided in the question.
    """
    return {
        'annotation': {
            'folder': 'images',
            'filename': 'image_0000005.jpg',
            'size': {
                'width': '1904',
                'height': '2856',
                'depth': '3'
            },
            'object': [
                {
                    'name': 'defect',
                    'difficult': '0',
                    'bndbox': {
                        'xmin': '661',
                        'ymin': '472',
                        'xmax': '992',
                        'ymax': '1857'
                    },
                    'Defect': {
                        'Background': '0',
                        'Crack': '0',
                        'Spallation': '0',
                        'Efflorescence': '1',
                        'ExposedBars': '0',
                        'CorrosionStain': '1'
                    }
                },
                {
                    'name': 'defect',
                    'difficult': '0',
                    'bndbox': {
                        'xmin': '1507',
                        'ymin': '505',
                        'xmax': '1904',
                        'ymax': '2856'
                    },
                    'Defect': {
                        'Background': '0',
                        'Crack': '0',
                        'Spallation': '0',
                        'Efflorescence': '1',
                        'ExposedBars': '0',
                        'CorrosionStain': '1'
                    }
                }
            ]
        }
    }


def test_parse_dict_annotation(sample_xml_dict):
    """
    Test that parse_dict_annotation correctly converts the sample XML dict
    to a Pandas DataFrame with expected values.
    """
    parser = AnnotationsParser(folder_path="fake_folder")

    df = parser.parse_dict_annotation(sample_xml_dict)

    # We have two objects => 2 rows
    assert len(df) == 2

    # Check columns
    expected_cols = [
        'img_name', 'xmin', 'ymin', 'xmax', 'ymax',
        'Background', 'Crack', 'Spallation', 'Efflorescence',
        'ExposedBars', 'CorrosionStain'
    ]
    assert all(col in df.columns for col in expected_cols)

    # Check the image name
    assert (df['img_name'] == 'image_0000005.jpg').all()

    # Check bounding box columns
    # First row
    assert df.loc[0, 'xmin'] == 661
    assert df.loc[0, 'ymin'] == 472
    assert df.loc[0, 'xmax'] == 992
    assert df.loc[0, 'ymax'] == 1857

    # Second row
    assert df.loc[1, 'xmin'] == 1507
    assert df.loc[1, 'ymin'] == 505
    assert df.loc[1, 'xmax'] == 1904
    assert df.loc[1, 'ymax'] == 2856

    # Check defect columns
    # For both rows: Background=0, Crack=0, Spallation=0, Efflorescence=1, ExposedBars=0, CorrosionStain=1
    assert (df['Background'] == 0).all()
    assert (df['Crack'] == 0).all()
    assert (df['Spallation'] == 0).all()
    assert (df['Efflorescence'] == 1).all()
    assert (df['ExposedBars'] == 0).all()
    assert (df['CorrosionStain'] == 1).all()


def test_parse_dict_annotation_no_defects():
    """
    Test that parse_dict_annotation returns a single-row DataFrame with Background=1
    if no <object> with name='defect' is found.
    """
    sample_dict_no_defects = {
        'annotation': {
            'folder': 'images',
            'filename': 'image_0000010.jpg',
            'size': {
                'width': '1000',
                'height': '1000',
                'depth': '3'
            },
            'object': [
                {
                    'name': 'non_defect',
                    'difficult': '0'
                }
            ]
        }
    }
    parser = AnnotationsParser(folder_path="fake_folder")
    df = parser.parse_dict_annotation(sample_dict_no_defects)

    assert len(df) == 1
    assert df.loc[0, 'img_name'] == 'image_0000010.jpg'
    assert df.loc[0, 'Background'] == 1
    # All other defect columns should be 0
    assert df.loc[0, 'Crack'] == 0
    assert df.loc[0, 'Spallation'] == 0
    assert df.loc[0, 'Efflorescence'] == 0
    assert df.loc[0, 'ExposedBars'] == 0
    assert df.loc[0, 'CorrosionStain'] == 0


def test_create_background_dict():
    """
    Test that create_Background_dict creates a dictionary with
    Background=1 and other defect columns set to 0.
    """
    parser = AnnotationsParser(folder_path="fake_folder")
    out_dict = parser.create_background_dict("image_0000011.jpg")

    assert out_dict['img_name'] == 'image_0000011.jpg'
    assert out_dict['Background'] == 1
    assert out_dict['Crack'] == 0
    assert out_dict['Spallation'] == 0
    assert out_dict['Efflorescence'] == 0
    assert out_dict['ExposedBars'] == 0
    assert out_dict['CorrosionStain'] == 0


def test_fill_df_with_missing_images():
    """
    Test that fill_df_with_missing_images adds rows for missing images,
    setting them as Background=1.
    """
    # Print the class to confirm correct import
    print(AnnotationsParser)  # Should output: <class 'src.preprocessing.AnnotationsParser.AnnotationsParser'>

    # Initialize the AnnotationsParser
    parser = AnnotationsParser(folder_path="fake_folder")
    assert parser is not None

    # Prepare a DataFrame with intentional missing images
    # For example, images 1 and 4 are present; 2 and 3 are missing
    data = {
        'img_name': ['image0000001.jpg', 'image0000004.jpg'],
        'xmin': [0, 0],
        'ymin': [0, 0],
        'xmax': [0, 0],
        'ymax': [0, 0],
        'Background': [0, 0],
        'Crack': [0, 0],
        'Spallation': [0, 0],
        'Efflorescence': [1, 1],
        'ExposedBars': [0, 0],
        'CorrosionStain': [1, 1]
    }
    df = pd.DataFrame(data)

    # Mock 'create_background_dict' to return scalar values with '.jpg' in img_name
    with patch.object(parser, 'create_background_dict', side_effect=lambda x: {
        'img_name': x,  # Ensure '.jpg' is included
        'xmin': 0,
        'ymin': 0,
        'xmax': 0,
        'ymax': 0,
        'Background': 1,
        'Crack': 0,
        'Spallation': 0,
        'Efflorescence': 0,
        'ExposedBars': 0,
        'CorrosionStain': 0
    }):
        # Call the method under test
        new_df = parser.fill_df_with_missing_images(df)

    # Assertions to verify that missing images are added correctly
    # Original df has 2 images; expect 2 more to be added for a total of 4
    assert len(new_df) == 4, f"Expected 4 rows, got {len(new_df)}"

    # Verify that the correct missing images are added
    expected_missing_imgs = {'image0000002.jpg', 'image0000003.jpg'}
    actual_missing_imgs = set(new_df['img_name']) - set(df['img_name'])
    assert expected_missing_imgs == actual_missing_imgs, (
        f"Expected missing images {expected_missing_imgs}, got {actual_missing_imgs}"
    )



@patch('os.listdir')
def test_parse_xmls_to_dataframe(mock_listdir, sample_xml_dict):
    """
    Test that parse_xmls_to_dataframe reads all .xml files in folder
    and returns the expected DataFrame.
    """
    # Mock the directory contents
    mock_listdir.return_value = ["annotation1.xml", "annotation2.txt", "annotation3.xml"]

    # We will mock parse_xml_to_dict so that it returns sample_xml_dict for .xml files
    parser = AnnotationsParser(folder_path="fake_folder")
    with patch.object(parser, 'parse_xml_to_dict') as mock_parse:
        def side_effect(filepath):
            if filepath.endswith(".xml"):
                return sample_xml_dict
            return None

        mock_parse.side_effect = side_effect

        df = parser.parse_xmls_to_dataframe()

        # parse_xmls_to_dataframe should parse 2 XML files
        # Each mock XML is the same sample_xml_dict => 2 objects each => 2 rows each => 4 total
        assert len(df) == 4
        # Efflorescence and CorrosionStain are both 1
        assert (df['Efflorescence'] == 1).all()
        assert (df['CorrosionStain'] == 1).all()

        # Also check that non-XML file was skipped
        assert mock_parse.call_count == 2  # called for each file in listdir
        # The data for annotation2.txt should have returned None
        # The data for annotation1.xml and annotation3.xml should match sample_xml_dict


################################################################################
#               ADDITIONAL TESTS (from screenshot scenarios)                   #
################################################################################

@pytest.mark.parametrize(
    "file_contents_dict",
    [
        {
            "sample_xml_multiple_objects.xml": """<?xml version='1.0'?>
            <annotation>
              <filename>image_000001.jpg</filename>
              <object>
                <name>defect</name>
                <bndbox><xmin>10</xmin><ymin>20</ymin><xmax>30</xmax><ymax>40</ymax></bndbox>
                <Defect>
                  <Crack>1</Crack>
                  <Efflorescence>0</Efflorescence>
                  <Background>0</Background>
                </Defect>
              </object>
              <object>
                <name>defect</name>
                <bndbox><xmin>50</xmin><ymin>60</ymin><xmax>70</xmax><ymax>80</ymax></bndbox>
                <Defect>
                  <Spallation>1</Spallation>
                  <CorrosionStain>1</CorrosionStain>
                  <Background>0</Background>
                </Defect>
              </object>
            </annotation>
            """
        }
    ],
)
def test_parse_multiple_objects(file_contents_dict):
    """
    Test parsing an XML file containing multiple <object> entries.
    """
    parser = AnnotationsParser(folder_path="fake_folder")
    with patch("os.listdir", new=mock_listdir_factory(file_contents_dict)):
        with patch("builtins.open", new=mock_open_factory(file_contents_dict)):
            df = parser.parse_xmls_to_dataframe()
            # We expect 2 rows for the 2 <object> entries
            assert len(df) == 2


@pytest.mark.parametrize(
    "file_contents_dict",
    [
        {
            "sample_xml_single_object.xml": """<?xml version='1.0'?>
            <annotation>
              <filename>image_000002.jpg</filename>
              <object>
                <name>defect</name>
                <bndbox><xmin>10</xmin><ymin>20</ymin><xmax>30</xmax><ymax>40</ymax></bndbox>
                <Defect><Crack>1</Crack><Background>0</Background></Defect>
              </object>
            </annotation>
            """
        }
    ],
)
def test_parse_single_object(file_contents_dict):
    parser = AnnotationsParser(folder_path="fake_folder")
    with patch("os.listdir", new=mock_listdir_factory(file_contents_dict)):
        with patch("builtins.open", new=mock_open_factory(file_contents_dict)):
            df = parser.parse_xmls_to_dataframe()
            assert len(df) == 1
            assert df.loc[0, 'Crack'] == 1


@pytest.mark.parametrize(
    "file_contents_dict",
    [
        {
            "sample_xml_no_objects.xml": """<?xml version='1.0'?>
            <annotation>
              <filename>image_000003.jpg</filename>
            </annotation>
            """
        }
    ],
)
def test_parse_no_objects(file_contents_dict):
    """
    This is similar to test_parse_dict_annotation_no_defects but
    checks the entire pipeline with parse_xmls_to_dataframe.
    """
    parser = AnnotationsParser(folder_path="fake_folder")
    with patch("os.listdir", new=mock_listdir_factory(file_contents_dict)):
        with patch("builtins.open", new=mock_open_factory(file_contents_dict)):
            df = parser.parse_xmls_to_dataframe()
            # With no <object> named 'defect', we expect 1 row with background=1
            assert len(df) == 1
            assert df.loc[0, 'Background'] == 1


@pytest.mark.parametrize(
    "file_contents_dict",
    [
        {
            "sample_xml_malformed.xml": """<?xml version='1.0'?>
            <annotation>
              <filename>image_000004.jpg</filename>
              <object>
                <name>defect</name>
            <!-- missing closing tags, etc. -->
            """
        }
    ],
)
def test_parse_malformed_xml(file_contents_dict):
    parser = AnnotationsParser(folder_path="fake_folder")
    with patch("os.listdir", new=mock_listdir_factory(file_contents_dict)):
        with patch("builtins.open", new=mock_open_factory(file_contents_dict)):
            # parse_xml_to_dict should return None or fail to parse,
            # so parse_xmls_to_dataframe should produce an empty or background-only df
            df = parser.parse_xmls_to_dataframe()
            assert df.empty


@pytest.mark.parametrize(
    "file_contents_dict",
    [
        {
            "sample_xml_missing_fields.xml": """<?xml version='1.0'?>
            <annotation>
              <filename>image_000005.jpg</filename>
              <object>
                <name>defect</name>
                <!-- missing bndbox entirely -->
                <Defect><Crack>1</Crack></Defect>
              </object>
            </annotation>
            """
        }
    ],
)
def test_parse_missing_fields(file_contents_dict, caplog):
    parser = AnnotationsParser(folder_path="fake_folder")
    with patch("os.listdir", new=mock_listdir_factory(file_contents_dict)):
        with patch("builtins.open", new=mock_open_factory(file_contents_dict)):
            df = parser.parse_xmls_to_dataframe()
            # Should produce 1 row. Bndbox coords might be missing => possibly NaN or 0.
            # Just check we didn't crash
            assert len(df) == 1


@pytest.mark.parametrize(
    "file_contents_dict",
    [
        {
            "sample_xml_invalid_defect_values.xml": """<?xml version='1.0'?>
            <annotation>
              <filename>image_000006.jpg</filename>
              <object>
                <name>defect</name>
                <bndbox><xmin>10</xmin><ymin>20</ymin><xmax>30</xmax><ymax>40</ymax></bndbox>
                <Defect><Crack>not_an_int</Crack></Defect>
              </object>
            </annotation>
            """
        }
    ],
)
def test_parse_invalid_defect_values(file_contents_dict, caplog):
    parser = AnnotationsParser(folder_path="fake_folder")
    with patch("os.listdir", new=mock_listdir_factory(file_contents_dict)):
        with patch("builtins.open", new=mock_open_factory(file_contents_dict)):
            df = parser.parse_xmls_to_dataframe()
            assert len(df) == 1
            # Should log a warning and set Crack=0
            assert df.loc[0, 'Crack'] == 0
            assert any("Invalid value for Crack" in rec.message for rec in caplog.records)


@pytest.mark.parametrize(
    "file_contents_dict",
    [
        # Provide multiple XML files to simulate large data or missing images
        {
            "sample_xml_single_object.xml": """<?xml version='1.0'?>
            <annotation>
              <filename>image_000010.jpg</filename>
              <object>
                <name>defect</name>
                <bndbox><xmin>10</xmin><ymin>20</ymin><xmax>30</xmax><ymax>40</ymax></bndbox>
                <Defect><Crack>1</Crack></Defect>
              </object>
            </annotation>
            """,
            "sample_xml_no_objects.xml": """<?xml version='1.0'?>
            <annotation>
              <filename>image_000011.jpg</filename>
            </annotation>
            """,
        }
    ],
)
def test_fill_missing_images_large(file_contents_dict):
    """
    Test that all images up to 1600 are considered, and any that
    aren't in the parsed results get background=1.
    """
    parser = AnnotationsParser(folder_path="fake_folder")
    with patch("os.listdir", new=mock_listdir_factory(file_contents_dict)):
        with patch("builtins.open", new=mock_open_factory(file_contents_dict)):
            df = parser.parse_xmls_to_dataframe()
            # The parser calls fill_df_with_missing_images in _initialize_annotations,
            # so you might have 1600 rows total, or at least background-filled for any missing.
            # Check at least that your known images are present:
            assert "image_000010.jpg" in df['img_name'].values
            assert "image_000011.jpg" in df['img_name'].values
            # The rest presumably are background. You can do further checks.


@pytest.mark.parametrize("file_contents_dict", [{}])
def test_no_xml_files_screenshot(file_contents_dict):
    """
    If the folder is empty or has no .xml files, parse_xmls_to_dataframe
    should return either an empty DataFrame or 1600 background rows,
    depending on your fill logic.
    """
    parser = AnnotationsParser(folder_path="fake_folder")
    with patch("os.listdir", new=mock_listdir_factory(file_contents_dict)):
        df = parser.parse_xmls_to_dataframe()
        # Depending on your code, might be empty or might have 1600 background rows.
        # Just ensure it doesn't crash.
        # For demonstration, we'll accept either case:
        assert df.shape[0] == 0 or df.shape[0] == 1600


@pytest.mark.parametrize(
    "file_contents_dict",
    [
        {
            "sample_xml_non_defect_object.xml": """<?xml version='1.0'?>
            <annotation>
              <filename>image_000012.jpg</filename>
              <object>
                <name>cat</name>
                <bndbox><xmin>0</xmin><ymin>0</ymin><xmax>10</xmax><ymax>10</ymax></bndbox>
              </object>
            </annotation>
            """
        }
    ],
)
def test_non_defect_objects_screenshot(file_contents_dict):
    parser = AnnotationsParser(folder_path="fake_folder")
    with patch("os.listdir", new=mock_listdir_factory(file_contents_dict)):
        with patch("builtins.open", new=mock_open_factory(file_contents_dict)):
            df = parser.parse_xmls_to_dataframe()
            # Should produce 1 row with background=1, since <object> is not a 'defect'
            assert len(df) == 1
            assert df.loc[0, 'Background'] == 1


@pytest.mark.parametrize(
    "file_contents_dict",
    [
        {
            "sample_xml_invalid_filename.xml": """<?xml version='1.0'?>
            <annotation>
              <filename></filename>
              <object>
                <name>defect</name>
                <bndbox><xmin>10</xmin><ymin>20</ymin><xmax>30</xmax><ymax>40</ymax></bndbox>
              </object>
            </annotation>
            """
        }
    ],
)
def test_invalid_image_filenames(file_contents_dict, caplog):
    parser = AnnotationsParser(folder_path="fake_folder")
    with patch("os.listdir", new=mock_listdir_factory(file_contents_dict)):
        with patch("builtins.open", new=mock_open_factory(file_contents_dict)):
            df = parser.parse_xmls_to_dataframe()
            # If filename is empty, you might store "" or log a warning. Adapt to your logic.
            assert len(df) == 1


@pytest.mark.parametrize(
    "file_contents_dict",
    [
        {
            "sample_xml_duplicate_filenames1.xml": """<?xml version='1.0'?>
            <annotation>
              <filename>image_000020.jpg</filename>
              <object><name>defect</name></object>
            </annotation>
            """,
            "sample_xml_duplicate_filenames2.xml": """<?xml version='1.0'?>
            <annotation>
              <filename>image_000020.jpg</filename>
              <object><name>defect</name></object>
            </annotation>
            """
        }
    ],
)
def test_duplicate_image_filenames(file_contents_dict, caplog):
    """
    Tests how duplicates are handled. Does your code combine them?
    Overwrite one? Create multiple rows for the same filename?
    """
    parser = AnnotationsParser(folder_path="fake_folder")
    with patch("os.listdir", new=mock_listdir_factory(file_contents_dict)):
        with patch("builtins.open", new=mock_open_factory(file_contents_dict)):
            df = parser.parse_xmls_to_dataframe()
            # Behavior is project-specific. Possibly 2 rows for the same image.
            assert not df.empty
            # E.g., if each file leads to a row, might have 2:
            # assert len(df) == 2


@pytest.mark.parametrize(
    "file_contents_dict",
    [
        # Provide many XML files to test reading capacity or performance
        {f"file_{i}.xml": f"""<?xml version='1.0'?><annotation><filename>image_{i}.jpg</filename></annotation>"""
         for i in range(1, 101)}  # 100 small XMLs
    ],
)
def test_large_number_of_xml_files(file_contents_dict):
    parser = AnnotationsParser(folder_path="fake_folder")
    with patch("os.listdir", new=mock_listdir_factory(file_contents_dict)):
        with patch("builtins.open", new=mock_open_factory(file_contents_dict)):
            df = parser.parse_xmls_to_dataframe()
            # Should produce 100 rows (1 for each file)
            assert len(df) == 100


@pytest.mark.parametrize(
    "file_contents_dict",
    [
        {
            "sample_xml_single_object.xml": """<?xml version='1.0'?>
            <annotation>
              <filename>image_0000030.jpg</filename>
              <object>
                <name>defect</name>
                <bndbox><xmin>10</xmin><ymin>20</ymin><xmax>30</xmax><ymax>40</ymax></bndbox>
                <Defect><Crack>1</Crack></Defect>
              </object>
            </annotation>
            """
        }
    ],
)
def test_annotations_df_property(file_contents_dict):
    """
    Test the `annotations_df` property, which should be populated after initialization.
    """
    with patch("os.listdir", new=mock_listdir_factory(file_contents_dict)):
        with patch("builtins.open", new=mock_open_factory(file_contents_dict)):
            parser = AnnotationsParser(folder_path="fake_folder")
            df = parser.annotations_df
            assert not df.empty
            assert "Crack" in df.columns


@pytest.mark.parametrize(
    "file_contents_dict",
    [
        {
            "sample_xml_non_defect_object.xml": """<?xml version='1.0'?>
            <annotation>
              <filename>image_0000031.jpg</filename>
              <object>
                <name>cat</name>
                <bndbox><xmin>0</xmin><ymin>0</ymin><xmax>10</xmax><ymax>10</ymax></bndbox>
              </object>
            </annotation>
            """
        }
    ],
)
def test_non_defect_objects_logging(file_contents_dict, caplog):
    """
    Check that ignoring non-defect objects is (optionally) logged.
    """
    with patch("os.listdir", new=mock_listdir_factory(file_contents_dict)):
        with patch("builtins.open", new=mock_open_factory(file_contents_dict)):
            parser = AnnotationsParser(folder_path="fake_folder")
            df = parser.parse_xmls_to_dataframe()
            assert len(df) == 1
            assert df.loc[0, 'Background'] == 1
            # Optionally check logs if your code logs ignoring messages:
            # assert any("Ignoring object" in rec.message for rec in caplog.records)
