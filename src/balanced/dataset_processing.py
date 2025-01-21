import os
from os import makedirs
from os.path import join, abspath
from sys import platform

from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import PIL.Image as Image

import src.utils as u
import src.constants as c


def resize_and_pad_yolo(image_path, output_path, target_size) -> list[float]:
    # Open an image file
    with Image.open(image_path) as img:
        # Get original dimensions
        original_width, original_height = img.size

        # Determine the scaling factor based on the larger dimension
        if original_width > original_height:
            scale_factor = target_size / original_width
        else:
            scale_factor = target_size / original_height

        # Calculate new dimensions
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)

        # Resize the image
        resized_img = img.resize((new_width, new_height))

        # Create a new square image with a black background
        square_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))

        # Calculate position to center the resized image
        position = ((target_size - new_width) // 2, (target_size - new_height) // 2)

        # Paste the resized image onto the square image
        square_img.paste(resized_img, position)

        # Save the padded image
        square_img.save(output_path)

        # Calculate YOLO format coordinates
        # Center of the bounding box
        center_x = (position[0] + new_width / 2) / target_size
        center_y = (position[1] + new_height / 2) / target_size

        # Width and height of the bounding box
        width = new_width / target_size
        height = new_height / target_size

        # Return the padded image and YOLO format coordinates
        return [center_x, center_y, width, height]


def copy_bg_and_defects(base_input_path, base_output_path) -> dict[str, str]:
    split_dirs = {
        'train': join(base_output_path, 'train'),
        'val': join(base_output_path, 'val'),
        'test': join(base_output_path, 'test')
    }

    if platform == 'linux' or platform == 'linux2':
        base_out_splitted = str(base_output_path).split('/')
    else:
        base_out_splitted = str(base_output_path).split('\\')

    for split in ['val', 'test', 'train']:
        # Create split directory
        split_dir = join(base_output_path, split)
        makedirs(split_dir, exist_ok=True)

        print(f'copying {split} defect imgs to ~/{join(*base_out_splitted[-3:], split)}')
        u.copy_imgs(abspath(join(base_input_path, split, 'defects')), split_dir)
        print(f'copying {split} background imgs to ~/{join(*base_out_splitted[-3:], split)}')
        u.copy_imgs(join(base_input_path, split, 'background'), split_dir)

    return split_dirs

def plot_corr_matrix(org_df: pd.DataFrame, figsize: tuple[int, int] = (10, 8)) -> None:
    # generated with Anthropic's Claude Sonnet 3.5
    cols_for_plot = list(org_df.select_dtypes(include=np.number).columns)
    # Create correlation matrix
    corr_matrix = np.zeros((len(cols_for_plot), len(cols_for_plot)))
    np.fill_diagonal(corr_matrix, 1.0)  # Fill diagonal with 1s

    # Calculate correlations
    for (i, col1), (j, col2) in combinations(enumerate(cols_for_plot), 2):
        correlation = org_df[col1].corr(org_df[col2])
        corr_matrix[i, j] = correlation
        corr_matrix[j, i] = correlation  # Make matrix symmetric


    # Convert to DataFrame for better visualization
    corr_df = pd.DataFrame(corr_matrix, columns=cols_for_plot, index=cols_for_plot)

    # Create the plot
    plt.figure(figsize=figsize)
    sns.heatmap(corr_df,
                annot=True,  # Show correlation values
                cmap='coolwarm',  # Red-blue colormap
                vmin=-1, vmax=1,  # Set correlation range
                center=0,  # Center the colormap at 0
                square=True,  # Make cells square
                fmt='.2f')  # Format correlation values to 2 decimal places

    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()

    # Print strongest correlations
    print("\nStrongest Correlations:")
    print("=======================")
    correlations = []
    for (i, col1), (j, col2) in combinations(enumerate(cols_for_plot), 2):
        correlations.append((col1, col2, corr_matrix[i, j]))

    # Sort by absolute correlation value
    sorted_correlations = sorted(correlations, key=lambda x: abs(x[2]), reverse=True)

    # Print top 5 strongest correlations
    for col1, col2, corr in sorted_correlations[:5]:
        print(f"{col1} - {col2}: {corr:.3f}")