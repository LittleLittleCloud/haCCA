
from collections import Counter
import os
from matplotlib import cm, pyplot as plt
import cv2
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from hacca.data import Data

color_mapping = {
        0: 'blue',
        1: 'red',
        2: 'green',
        3: 'yellow',
        4: 'orange',
        5: 'purple',
        6: 'brown',
        7: 'pink',
        8: 'gray',
        9: 'cyan',
        10: 'magenta',
        11: 'cyan',
        12: 'blue',
        13: 'red',
        14: 'green',
        15: 'yellow'
    }

def count_elements(lst):
    return dict(Counter(lst))

def calculate_simpson_index(values):
    total_count = len(values)
    unique_values = set(values)
    counters = count_elements(values)
    simpson_index = 1 - sum((counters[value] * 1.0 / total_count) ** 2 for value in unique_values)
    return simpson_index

def create_image_from_data(data, width=500, height=500, dot_size=5, border_size=50, dot_colors=None, border_color=(0, 0, 0), colormap='viridis'):
    """
    Create an image from 2D data points with specified dot size, border size, and colors.

    Parameters:
    - data: 2D numpy array where data[0] contains x coordinates and data[1] contains y coordinates
    - width: Width of the image
    - height: Height of the image
    - dot_size: Size of the dots
    - border_size: Size of the border around the image
    - dot_colors: List of scalar values for each dot, which will be mapped to colors using a colormap
    - border_color: Color of the border (BGR format)
    - colormap: Colormap to use for mapping scalar values to colors
    
    Returns:
    - bordered_image: Image with data points and border
    """
    # Initialize an empty image with three channels for color
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Normalize data points to image dimensions
    x_norm = np.interp(data[0], (data[0].min(), data[0].max()), (0, width-1))
    y_norm = np.interp(data[1], (data[1].min(), data[1].max()), (0, height-1))

    # Normalize dot_colors to [0, 1] range
    if dot_colors is not None:
        norm = plt.Normalize(vmin=min(dot_colors), vmax=max(dot_colors))
        cmap = cm.get_cmap(colormap)
        mapped_colors = cmap(norm(dot_colors))

    # Draw circles at normalized positions
    for i, (x, y) in enumerate(zip(x_norm, y_norm)):
        if dot_colors is not None:
            color = (mapped_colors[i][:3] * 255).astype(int)  # Convert to BGR format
            color = tuple(map(int, color[::-1]))  # Convert from RGB to BGR
        else:
            color = (255, 255, 255)  # Default white color
        cv2.circle(image, (int(x), int(y)), dot_size, color, -1)  # -1 means filled circle

    # Add border
    bordered_image = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, 
                                        cv2.BORDER_CONSTANT, value=border_color)

    return bordered_image

def click_event_image(event, x, y, points_image, display_image):
    if event == cv2.EVENT_LBUTTONDOWN:
        points_image.append((x, y))
        cv2.circle(display_image, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('Image', display_image)

def compute_affine_transform(points_image1, points_image2):
    pts1 = np.float32(points_image1)
    pts2 = np.float32(points_image2)
    M = cv2.getAffineTransform(pts2[:3], pts1[:3])  # Use the first 3 points for affine transform
    return M

def warp_image(image, M, output_shape):
    warped_image = cv2.warpAffine(image, M, output_shape)
    return warped_image

def apply_affine_transform_2d(coordinates, transformation_matrix):
    """
    对二维坐标应用仿射变换
    """
    # 添加齐次坐标
    homogeneous_coordinates = np.hstack([coordinates, np.ones((coordinates.shape[0], 1))])
    
    # 应用仿射变换
    transformed_coordinates = np.dot(homogeneous_coordinates, transformation_matrix.T)
    
    return transformed_coordinates[:, :2]


def center_and_scale(data, feature_range=(0, 500)):
    """
    对数据进行中心缩放和范围缩放
    """
    # 中心缩放
    scaler_center = StandardScaler()
    data_centered = scaler_center.fit_transform(data)
    
    # 范围缩放
    scaler_range = MinMaxScaler(feature_range=feature_range)
    data_scaled = scaler_range.fit_transform(data_centered)
    
    return data_scaled

def _generate_mock_data(n: int, feature_n: int, labels):
    """
    Generate [n, feature_n] data and [n, 2] distance matrix with random selection of labels
    """

    X = np.random.rand(n, feature_n)
    D = np.random.rand(n, 2)
    label_indices = np.random.randint(0, len(labels), n)
    labels = np.array([labels[i] for i in label_indices])

    return Data(X=X, D=D, Label=labels)

def _plot_alignment_result(a: Data, b_prime: Data, b_align: Data, pi, title="Transport Plan", work_dir=None, show=True):
    """
    Plot the transport plan from source to target
    It will plot two subplots, with left subplot showing the source data and right subplot showing the target data.
    For every spot in the source data, it will draw a line to the target data.
    - Left subplot: The source data
    - Right subplot: The target data

    source: Data, the source data
    target: Data, the aligned target data. It should have the same number of data points as the source data.
    """

    assert b_align.D.shape == b_prime.D.shape, "The aligned target data should have the same number of data points as the target data"
    assert b_align.X.shape[1] == a.X.shape[1], "The aligned target data should have the same number of features as the target data"

    # find out the max_x, min_x, max_y, min_y in the source and target data
    max_x = max(max(a.D[:, 0]), max(b_prime.D[:, 0]))
    min_x = min(min(a.D[:, 0]), min(b_prime.D[:, 0]))
    max_y = max(max(a.D[:, 1]), max(b_prime.D[:, 1]))
    min_y = min(min(a.D[:, 1]), min(b_prime.D[:, 1]))
    max_x = max_x + 0.1 * (max_x - min_x)
    min_x = min_x - 0.1 * (max_x - min_x)
    max_y = max_y + 0.1 * (max_y - min_y)
    min_y = min_y - 0.1 * (max_y - min_y)
    
    # Create a new figure
    fig = plt.figure(figsize=(12, 6))

    # Plot the source data and add labels
    source_label = a.Label.astype(int)
    source_label_colors = [color_mapping[label] for label in source_label]
    ax1 = plt.subplot(1, 2, 1)
    plt.scatter(a.D[:, 0], a.D[:, 1], c=source_label_colors, s=20)
    plt.title("A")
    ax1.set_xlim([min_x, max_x])
    ax1.set_ylim([min_y, max_y])


    # Plot the target data
    target_label = b_prime.Label.astype(int)
    target_label_colors = [color_mapping[label] for label in target_label]
    ax2 = plt.subplot(1, 2, 2)
    plt.scatter(b_prime.D[:, 0], b_prime.D[:, 1], c=target_label_colors, s=20)
    plt.title("B_Prime")
    ax2.set_xlim([min_x, max_x])
    ax2.set_ylim([min_y, max_y])

    # Plot the alignment lines
    transFigure = fig.transFigure.inverted()
    
    for i in range(b_prime.D.shape[0]):
        coord1 = transFigure.transform(ax1.transData.transform([b_align.D[i, 0], b_align.D[i, 1]]))
        coord2 = transFigure.transform(ax2.transData.transform([b_prime.D[i, 0], b_prime.D[i, 1]]))
        line = plt.Line2D((coord1[0], coord2[0]), (coord1[1], coord2[1]), transform=fig.transFigure, c='black', alpha=0.1)
        fig.lines.append(line)
        

    # Plot the cluster labels
    # unique_categories = np.unique(source_label)
    # handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_mapping[cat], markersize=10) for cat in unique_categories]
    # plt.legend(handles, unique_categories, title="Metabolic Clusters")

    if work_dir is not None:
        plt.savefig(os.path.join(work_dir, 'transport_plan.pdf'))
        
    if show is True:
        plt.show()
    
    return plt

def _plot_b_predict(
        b_prime: Data, # B' (X_2, D)
        work_dir: str = None, # the working directory, will be created if not exists. Will be used to save the intermediate results.
        save: bool = True
):
    b_prime_D = b_prime.D
    b_prime_labels = b_prime.Label.astype(int)

    if work_dir is not None:
        # 可视化
        plt.figure(figsize=(6, 6))
        plt.scatter(pd.DataFrame(b_prime_D).iloc[:, 0], pd.DataFrame(b_prime_D).iloc[:, 1],
                    c=[color_mapping[category] for category in b_prime_labels.tolist()], s=20, alpha=1)
        plt.ylabel('Y')
        unique_categories =  np.unique(b_prime_labels)
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_mapping[cat], markersize=10) for
                   cat in unique_categories]
        plt.legend(handles, unique_categories, title="Metabolic Clusters")
       # plt.show()
        if save is True:
            plt.savefig(os.path.join(work_dir, 'align.pdf'))
            
