import pytest
import pandas as pd
import numpy as np

from src.preprocessing import (
    transform_coordinates,
    transform_bounding_box,
    change_df_format
)


# -------------------------------
# Tests for transform_coordinates
# -------------------------------

def test_transform_coordinates_zero_angle():
    """
    Test that transform_coordinates returns the same coordinates
    if the angle is zero.
    """
    x, y = 10, 20
    angle = 0.0
    x_prim, y_prim = transform_coordinates(x, y, angle, angle_in_radians=False)
    assert x_prim == x, f"Expected x_prim={x}, got {x_prim}"
    assert y_prim == y, f"Expected y_prim={y}, got {y_prim}"


def test_transform_coordinates_angle_in_degrees():
    """
    Test that transform_coordinates handles angle in degrees correctly.
    Here we rotate a point (10,0) by 90 degrees around the origin.
    The result should be (0,10).
    """
    x, y = 10, 0
    angle = 90.0
    x_prim, y_prim = transform_coordinates(x, y, angle, angle_in_radians=False)
    assert x_prim == 0, f"Expected x_prim=0, got {x_prim}"
    assert y_prim == 10, f"Expected y_prim=10, got {y_prim}"


def test_transform_coordinates_angle_in_radians():
    """
    Test that transform_coordinates handles angle already in radians.
    Rotate the point (10,0) by pi/2 radians; expect (0,10).
    """
    x, y = 10, 0
    angle = np.pi / 2
    x_prim, y_prim = transform_coordinates(x, y, angle, angle_in_radians=True)
    assert x_prim == 0, f"Expected x_prim=0, got {x_prim}"
    assert y_prim == 10, f"Expected y_prim=10, got {y_prim}"


def test_transform_coordinates_negative_x_raises_value_error():
    """
    Rotating certain points/angles could produce negative x_prim.
    Check that it raises ValueError as specified.
    """
    # A 180-degree rotation of (10,0) => (-10,0)
    x, y = 10, 0
    angle = 180.0
    with pytest.raises(ValueError) as exc_info:
        transform_coordinates(x, y, angle, angle_in_radians=False)
    assert "x coordinate cannot be negative" in str(exc_info.value)


def test_transform_coordinates_negative_y_raises_value_error():
    """
    Similar check for negative y_prim.
    """
    # A 180-degree rotation of (0,10) => (0,-10)
    x, y = 0, 10
    angle = 180.0
    with pytest.raises(ValueError) as exc_info:
        transform_coordinates(x, y, angle, angle_in_radians=False)
    assert "y coordinate cannot be negative" in str(exc_info.value)


# -------------------------------
# Tests for transform_bounding_box
# -------------------------------

def test_transform_bounding_box_no_rotation():
    """
    If angle is zero, we expect the same bounding box corners.
    """
    xmin, xmax, ymin, ymax = 10, 20, 30, 40
    angle = 0.0
    angle_in_radians = False
    result = transform_bounding_box(xmin, xmax, ymin, ymax, angle, angle_in_radians)
    expected = np.array([10, 30, 10, 40, 20, 40, 20, 30])  # (x1,y1,x2,y2,x3,y3,x4,y4)
    assert np.array_equal(result, expected), f"Expected {expected}, got {result}"


def test_transform_bounding_box_with_rotation():
    """
    Test a non-zero rotation. Because transform_coordinates can raise an error if
    negative coordinates occur, we use small angles to avoid negative results.

    We'll use a bounding box with all corners in quadrant 1 and rotate by 90 deg.
    """
    xmin, xmax, ymin, ymax = 10, 20, 10, 20
    angle = 90.0
    angle_in_radians = False

    # After a 90-degree rotation about the origin:
    # (10,10) -> (-10,10) => ValueError because x becomes negative
    # So let's do a smaller angle that keeps everything positive, say 45 degrees.

    angle = 45.0
    result = transform_bounding_box(xmin, xmax, ymin, ymax, angle, angle_in_radians)

    # We'll just check the shape and the positivity,
    # because an exact numeric check can be fiddly due to rounding.
    assert result.shape == (8,), "Expected array of length 8"
    assert np.all(result >= 0), f"Expected all values to be >= 0, got {result}"


def test_transform_bounding_box_raises_value_error_for_negative():
    """
    If the bounding box corners go negative, we should see a ValueError.
    Let's create a bounding box in quadrant 1 but rotate it 180 deg => negative corners.
    """
    xmin, xmax, ymin, ymax = 10, 20, 10, 20
    angle = 180.0
    angle_in_radians = False
    with pytest.raises(ValueError) as exc_info:
        transform_bounding_box(xmin, xmax, ymin, ymax, angle, angle_in_radians)
    assert "coordinate cannot be negative" in str(exc_info.value)


# -------------------------------
# Tests for change_df_format
# -------------------------------

def test_change_df_format_basic():
    """
    Check that change_df_format correctly converts columns
    and adds the new 8 columns, while keeping original data.
    """
    df = pd.DataFrame({
        'id': [1, 2],
        'xmin': [10, 50],
        'xmax': [20, 60],
        'ymin': [30, 70],
        'ymax': [40, 80]
    })

    new_df = change_df_format(df)

    # Check that original columns (besides xmin/xmax/ymin/ymax) are still present
    assert 'id' in new_df.columns, "Expected 'id' column to be in new_df"

    # Check that new columns are present
    for col in ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']:
        assert col in new_df.columns, f"Expected '{col}' to be in new_df"

    # Check expected shape: original (2 rows, plus old columns minus 4, plus new 8 columns)
    # Original: columns = ['id', 'xmin', 'xmax', 'ymin', 'ymax'] => total 5
    # We drop 4 from old, keep 1 => 1 old column + 8 new columns = 9 total => shape is (2, 9)
    assert new_df.shape == (2, 9), f"Expected shape (2, 9), got {new_df.shape}"

    # Verify the bounding box is unrotated => direct mapping
    # For the first row: (xmin=10, xmax=20, ymin=30, ymax=40)
    # => (x1=10,y1=30,x2=10,y2=40,x3=20,y3=40,x4=20,y4=30)
    assert all(new_df.loc[0, ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']] == [10, 30, 10, 40, 20, 40, 20, 30])


def test_change_df_format_empty_df():
    """
    Ensure it works gracefully with an empty DataFrame that has the required columns.
    """
    df = pd.DataFrame(columns=['xmin','xmax','ymin','ymax'])
    new_df = change_df_format(df)
    # Should contain the 8 new bbox columns, but no rows
    for col in ['x1','y1','x2','y2','x3','y3','x4','y4']:
        assert col in new_df.columns, f"Missing {col}"
    assert len(new_df) == 0, "Expected empty DataFrame after conversion."


def test_change_df_format_additional_columns():
    """
    If the input DF has additional columns (like 'label', 'score'),
    ensure they are preserved in the output.
    """
    df = pd.DataFrame({
        'label': ['cat', 'dog'],
        'score': [0.9, 0.8],
        'xmin': [1, 5],
        'xmax': [2, 6],
        'ymin': [10, 50],
        'ymax': [20, 60]
    })

    new_df = change_df_format(df)
    # Check if 'label' and 'score' still exist
    assert 'label' in new_df.columns, "Expected 'label' to be in new_df"
    assert 'score' in new_df.columns, "Expected 'score' to be in new_df"
    # Check shape: original had 6 columns, we drop 4, keep 2 => plus 8 new => total 10 columns
    assert new_df.shape == (2, 10), f"Expected shape (2, 10), got {new_df.shape}"
