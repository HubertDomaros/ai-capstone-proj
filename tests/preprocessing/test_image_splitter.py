import pytest
import pandas as pd
import numpy as np
import json
import os
from unittest import mock
from unittest.mock import patch, mock_open, MagicMock

# Adjust the import path based on your project structure
from src.preprocessing.image_splitter import (
    train_test_val_image_split,
    dict_to_json,
    put_splited_images_in_folders,
    raise_exeption_if_file_exists
)

# ----------------------------
# Tests for train_test_val_image_split
# ----------------------------

def test_train_test_val_image_split_basic():
    # Create sample DataFrame
    data = {
        'image_path': ['img1.jpg', 'img2.jpg', 'img1.jpg', 'img3.jpg'],
        'xmin': [0, 10, 5, 20],
        'xmax': [100, 110, 105, 120],
        'ymin': [0, 15, 10, 25],
        'ymax': [200, 210, 205, 220],
        'label1': [1, 0, 1, 0],
        'label2': [0, 1, 1, 0],
    }
    df = pd.DataFrame(data)

    # Expected grouping by 'image_path' with max labels
    expected_images = np.array(['img1.jpg', 'img2.jpg', 'img3.jpg'])

    # Mock iterative_train_test_split to return predefined splits
    with patch('src.preprocessing.image_splitter.iterative_train_test_split') as mock_split:
        # First split: train and rem
        mock_split.side_effect = [
            (np.array(['img1.jpg']), np.array([], dtype='<U8'), np.array(['img2.jpg', 'img3.jpg']), np.array([[0, 1], [0, 0]])),
            (np.array(['img2.jpg']), np.array([], dtype='<U8'), np.array(['img3.jpg']), np.array([[0, 0]]))
        ]

        result = train_test_val_image_split(df, test_size=0.4, val_size=0.1)

        # Assertions
        assert 'train' in result
        assert 'test' in result
        assert 'val' in result
        np.testing.assert_array_equal(result['train'], np.array(['img1.jpg']))
        np.testing.assert_array_equal(result['test'], np.array(['img2.jpg']))
        np.testing.assert_array_equal(result['val'], np.array(['img3.jpg']))
        # Check that iterative_train_test_split was called twice
        assert mock_split.call_count == 2


def test_train_test_val_image_split_empty_df():
    df = pd.DataFrame(columns=['image_path', 'xmin', 'xmax', 'ymin', 'ymax', 'label1', 'label2'])

    with pytest.raises(ValueError, match='Input DataFrame is empty'):
        train_test_val_image_split(df)


def test_train_test_val_image_split_single_image():
    data = {
        'image_path': ['img1.jpg', 'img1.jpg'],
        'xmin': [0, 5],
        'xmax': [100, 105],
        'ymin': [0, 10],
        'ymax': [200, 205],
        'label1': [1, 1],
        'label2': [0, 1],
    }
    df = pd.DataFrame(data)

    with patch('src.preprocessing.image_splitter.iterative_train_test_split') as mock_split:
        # First split: all data to train, none to remaining
        # Second split: empty remaining
        mock_split.side_effect = [
            (
                np.array(['img1.jpg']),
                np.array([], dtype='<U8'),
                np.array([], dtype='<U8'),
                np.array([[1, 1]])
            ),
            (
                np.array([], dtype='<U8'),
                np.array([], dtype='<U8'),
                np.array([], dtype='<U8'),
                np.array([])
            )
        ]

        result = train_test_val_image_split(df)

        # Assertions
        assert 'train' in result
        assert 'test' in result
        assert 'val' in result
        np.testing.assert_array_equal(result['train'], np.array(['img1.jpg']))
        np.testing.assert_array_equal(result['test'], np.array([], dtype='<U8'))
        np.testing.assert_array_equal(result['val'], np.array([], dtype='<U8'))
        # Check that iterative_train_test_split was called twice
        assert mock_split.call_count == 2


def test_train_test_val_image_split_invalid_sizes():
    df = pd.DataFrame({
        'image_path': ['img1.jpg', 'img2.jpg'],
        'xmin': [0, 10],
        'xmax': [100, 110],
        'ymin': [0, 15],
        'ymax': [200, 210],
        'label1': [1, 0],
        'label2': [0, 1],
    })

    with pytest.raises(ValueError, match="The sum of test_size and val_size must be less than 1."):
        # Sum of test_size and val_size > 1
        train_test_val_image_split(df, test_size=0.7, val_size=0.4)

# ----------------------------
# Tests for dict_to_json
# ----------------------------

def test_dict_to_json_success(tmp_path):
    input_dict = {
        'train': np.array(['img1.jpg', 'img2.jpg']),
        'test': np.array(['img3.jpg']),
        'val': np.array(['img4.jpg'])
    }
    out_filepath = tmp_path / "output.json"

    dict_to_json(input_dict, str(out_filepath))

    # Read the file and verify content
    with open(out_filepath, 'r') as f:
        data = json.load(f)

    assert data == {
        'train': ['img1.jpg', 'img2.jpg'],
        'test': ['img3.jpg'],
        'val': ['img4.jpg']
    }


def test_dict_to_json_empty_dict(tmp_path):
    input_dict = {}
    out_filepath = tmp_path / "empty.json"

    dict_to_json(input_dict, str(out_filepath))

    with open(out_filepath, 'r') as f:
        data = json.load(f)

    assert data == {}

# ----------------------------
# Tests for put_splited_images_in_folders
# ----------------------------

@patch('src.preprocessing.image_splitter.os.symlink')
@patch('src.preprocessing.image_splitter.os.makedirs')
@patch('src.preprocessing.image_splitter.os.path.exists')
def test_put_splited_images_in_folders_success(mock_exists, mock_makedirs, mock_symlink, tmp_path):
    # Setup mock JSON data
    json_data = {
        'train': ['img1.jpg', 'img2.jpg'],
        'test': ['img3.jpg'],
        'val': ['img4.jpg']
    }
    json_str = json.dumps(json_data)

    # Mock open to read the JSON data
    m = mock_open(read_data=json_str)
    with patch('builtins.open', m):
        input_dir = "/input/images"
        out_dir = str(tmp_path / "output")
        json_filepath = "/path/to/splits.json"

        # Mock os.path.exists to return False for directories and files
        mock_exists.return_value = False

        put_splited_images_in_folders(json_filepath, input_dir, out_dir)

        # Check that makedirs was called for each dataset
        expected_calls = [
            mock.call(os.path.join(out_dir, 'train')),
            mock.call(os.path.join(out_dir, 'test')),
            mock.call(os.path.join(out_dir, 'val'))
        ]
        mock_makedirs.assert_has_calls(expected_calls, any_order=True)

        # Check that symlink was called correctly
        expected_symlink_calls = [
            mock.call(os.path.join(input_dir, 'img1.jpg'), os.path.join(out_dir, 'train', 'img1.jpg')),
            mock.call(os.path.join(input_dir, 'img2.jpg'), os.path.join(out_dir, 'train', 'img2.jpg')),
            mock.call(os.path.join(input_dir, 'img3.jpg'), os.path.join(out_dir, 'test', 'img3.jpg')),
            mock.call(os.path.join(input_dir, 'img4.jpg'), os.path.join(out_dir, 'val', 'img4.jpg')),
        ]
        mock_symlink.assert_has_calls(expected_symlink_calls, any_order=True)


@patch('src.preprocessing.image_splitter.os.path.exists')
@patch('src.preprocessing.image_splitter.os.makedirs')
@patch('src.preprocessing.image_splitter.os.symlink')
def test_put_splited_images_in_folders_existing_directory(mock_symlink, mock_makedirs, mock_exists, tmp_path):
    # Setup mock JSON data
    json_data = {
        'train': ['img1.jpg'],
        'test': [],
        'val': []
    }
    json_str = json.dumps(json_data)

    # Mock open to read the JSON data
    m = mock_open(read_data=json_str)
    with patch('builtins.open', m):
        input_dir = "/input/images"
        out_dir = str(tmp_path / "output")
        json_filepath = "/path/to/splits.json"

        # Simulate that 'train' directory already exists
        def exists_side_effect(path):
            return path == os.path.join(out_dir, 'train')

        mock_exists.side_effect = exists_side_effect

        with pytest.raises(OSError) as exc_info:
            put_splited_images_in_folders(json_filepath, input_dir, out_dir)

        expected_error = f'Cannot make directory, {os.path.join(out_dir, "train")} already exists'
        assert expected_error in str(exc_info.value)


@patch('src.preprocessing.image_splitter.os.symlink')
@patch('src.preprocessing.image_splitter.os.makedirs')
@patch('src.preprocessing.image_splitter.os.path.exists')
def test_put_splited_images_in_folders_existing_files(mock_exists, mock_makedirs, mock_symlink, tmp_path):
    # Setup mock JSON data
    json_data = {
        'train': ['img1.jpg', 'img2.jpg'],
        'test': [],
        'val': []
    }
    json_str = json.dumps(json_data)

    # Mock open to read the JSON data
    m = mock_open(read_data=json_str)
    with patch('builtins.open', m):
        input_dir = "/input/images"
        out_dir = str(tmp_path / "output")
        json_filepath = "/path/to/splits.json"

        # Simulate that directories do not exist initially
        mock_exists.return_value = False

        # Simulate that 'img2.jpg' already exists in 'train'
        def exists_side_effect(path):
            if path == os.path.join(out_dir, 'train', 'img2.jpg'):
                return True
            return False

        mock_exists.side_effect = exists_side_effect

        with patch('builtins.print') as mock_print:
            put_splited_images_in_folders(json_filepath, input_dir, out_dir)

            # Check that symlink was called for img1.jpg
            mock_symlink.assert_any_call(
                os.path.join(input_dir, 'img1.jpg'),
                os.path.join(out_dir, 'train', 'img1.jpg')
            )

            # Check that symlink was NOT called for img2.jpg since it exists
            symlink_call = (
                os.path.join(input_dir, 'img2.jpg'),
                os.path.join(out_dir, 'train', 'img2.jpg')
            )
            assert symlink_call not in mock_symlink.call_args_list, \
                f"symlink was called with {symlink_call}, but it should not have been."

            # Check that a message was printed for the existing file
            mock_print.assert_any_call(
                f'File img2.jpg in path {os.path.join(out_dir, "train")} already exists. Skipping copying operation'
            )

            # Additionally, ensure that 'Created directory ...\train' was printed
            mock_print.assert_any_call(
                f"Created directory {os.path.join(out_dir, 'train')}"
            )

            # Ensure that 'Created directory ...\test' and 'Created directory ...\val' were printed
            mock_print.assert_any_call(
                f"Created directory {os.path.join(out_dir, 'test')}"
            )
            mock_print.assert_any_call(
                f"Created directory {os.path.join(out_dir, 'val')}"
            )


# ----------------------------
# Tests for raise_exeption_if_file_exists
# ----------------------------

def test_raise_exeption_if_file_exists_existing():
    with patch('src.preprocessing.image_splitter.os.path.exists') as mock_exists:
        mock_exists.return_value = True
        with pytest.raises(FileExistsError):
            raise_exeption_if_file_exists("existing_file.jpg")


def test_raise_exeption_if_file_exists_not_existing():
    with patch('src.preprocessing.image_splitter.os.path.exists') as mock_exists:
        mock_exists.return_value = False
        # Should not raise
        try:
            raise_exeption_if_file_exists("new_file.jpg")
        except FileExistsError:
            pytest.fail("raise_exeption_if_file_exists raised FileExistsError unexpectedly!")
