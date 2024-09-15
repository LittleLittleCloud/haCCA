import os
from typing import Tuple, Union
from sklearn.cross_decomposition import CCA
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import ot
from collections import Counter
from .data import Data
from .utils import center_and_scale, create_image_from_data
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import minimize

def count_elements(lst):
    return dict(Counter(lst))

def calculate_simpson_index(values):
    total_count = len(values)
    unique_values = set(values)
    counters = count_elements(values)
    simpson_index = 1 - sum((counters[value] * 1.0 / total_count) ** 2 for value in unique_values)
    return simpson_index

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


def create_image_from_data(
        data,
        width: int = 500,
        height: int = 500,
        dot_size: int = 5,
        border_size: int = 50,
        dot_colors: str = None,
        border_color=(0, 0, 0),
        colormap: str ='viridis'):
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
    x_norm = np.interp(data[0], (data[0].min(), data[0].max()), (0, width - 1))
    y_norm = np.interp(data[1], (data[1].min(), data[1].max()), (0, height - 1))

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


def manual_gross_alignment(
        a: Data, # A (n, X_1, D),
        b_prime: Data, # B' (m, X_2, D),
        border_size: int = 100,
        width: int = 500,
        height : int =500,
        dot_size: int = 4,
        work_dir: str = None,# the working directory, will be created if not exists. Will be used to save the intermediate results.

):
    def click_event_image1(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points_image1.append((x, y))
            cv2.circle(img1_display, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Image 1', img1_display)

    def click_event_image2(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points_image2.append((x, y))
            cv2.circle(img2_display, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Image 2', img2_display)

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

    dot_colors_1 = a.Label.astype(int)
    dot_colors_2 = b_prime.Label.astype(int)
    img1 = create_image_from_data(pd.DataFrame(a.D), width=width, height=height, dot_size=dot_size, border_size=border_size,
                                  dot_colors=dot_colors_1, border_color=(255, 0, 0), colormap='viridis')
    img2 = create_image_from_data(pd.DataFrame(b_prime.D), width=width, height=height, dot_size=dot_size, border_size=border_size,
                                  dot_colors=dot_colors_2, border_color=(0, 0, 255), colormap='viridis')
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title('Image 1')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.title('Image 2')

    points_image1 = []
    points_image2 = []
    img1_display = img1.copy()
    img2_display = img2.copy()
    cv2.imshow('Image 1', img1_display)
    cv2.imshow('Image 2', img2_display)
    cv2.setMouseCallback('Image 1', click_event_image1)
    cv2.setMouseCallback('Image 2', click_event_image2)
    while len(points_image1) < 3 or len(points_image2) < 3:
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    # Compute the transformation matrix
    M = compute_affine_transform(points_image1, points_image2)
    rows, cols = img1.shape[:2]
    aligned_image = warp_image(img2, M, (cols, rows))

    # Display the original and aligned images
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title('Original Image 1')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB))
    plt.title('Aligned Image 2')
    plt.savefig(os.path.join(work_dir, 'affine_translated.png'))

    M_homogeneous = np.vstack([M, [0, 0, 1]])
    coordinates_data2 = pd.DataFrame(b_prime.D)
    transformed_coordinates_data2 = apply_affine_transform_2d(coordinates_data2 + border_size, M_homogeneous)
    transformed_coordinates_data2 = pd.DataFrame(transformed_coordinates_data2 - border_size)

    if work_dir is not None:
        plt.figure(figsize=(5, 5))
        plt.scatter(a.D[:,0], a.D[:,1], color='red', label='A Points')
        plt.scatter(transformed_coordinates_data2[0], transformed_coordinates_data2[1], color='blue',
                label='B prime transfer Points')
        plt.legend()
        plt.savefig(os.path.join(work_dir, 'manual_align.png'))
    b_predict = Data(X=b_prime.X, D=transformed_coordinates_data2.values, Label=b_prime.Label)

    # run direct alignment with the DY1 and DY2
    return b_predict

def calculate_accuracy_for_pairwise_alignment(
        pairs: pd.DataFrame, # the aligned pairs of point pairs (data1.X, data1.Y, data2.X, data2.Y)
) -> float:
    same_values_mask_directmerge = pairs['data1_cluster'] == pairs['data2_cluster']
    accuracy_directmerge = sum(same_values_mask_directmerge) / len(same_values_mask_directmerge)
    print(f"Accuracy for pairwise alignment: {accuracy_directmerge}")
    return accuracy_directmerge



def further_alignment(
        a: Data, # A (X_1, D),
        b_prime: Data, # B' (X_2, D),
        work_dir: str = None, # the working directory, will be created if not exists. Will be used to save the intermediate results.
        dist_min: float = 2, # the minimum distance for further alignment
        verbose: bool = False
) -> Data: # b (n, [X_2], D)
    DF2 = pd.DataFrame(b_prime.D)
    DF3 = pd.DataFrame(a.D)
    Y_c = pd.DataFrame(b_prime.D)
    Z_c = pd.DataFrame(a.D)
    data2_leiden_str = b_prime.Label
    data1_leiden_str = a.Label

    def perform_transformations(DF2, DF3, Y_c, Z_c, data2_leiden_str, data1_leiden_str):
        transform_matrix = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        max_runs = 100
        run_count = 0
        while run_count < max_runs:
            run_count += 1
            distances = cdist(DF3, DF2, metric='euclidean')
            dist_df = pd.DataFrame(distances)

            row_indices, col_indices = np.where(distances < dist_min)
            unique_row_indices = np.unique(row_indices)
            unique_col_indices = np.unique(col_indices)

            Y_C_ = Y_c.iloc[col_indices, :]
            values = Y_C_.index
            Y_C_.loc[:,'BprimeID'] = values
            Y_C_ = Y_C_.reset_index()
            Y_C_.loc[:,"Bprime_label"] = data2_leiden_str[col_indices].astype(int)
            Y_C_.loc[:,"Bprime.Z"] = -1
            Y_C_ = Y_C_.drop(columns=['index'])
            Y_C_ = Y_C_.rename(columns={0: "Bprime.X", 1: "Bprime.Y"})

            Z_C_ = Z_c.iloc[row_indices, :]
            values = Z_C_.index
            Z_C_.loc[:,'AID'] = values
            Z_C_ = Z_C_.reset_index()
            Z_C_.loc[:,"A_label"] = data1_leiden_str[row_indices].astype(int)
            Z_C_ = Z_C_.drop(columns=['index'])
            Z_C_.loc[:,"A.Z"] = 1
            Z_C_ = Z_C_.rename(columns={0: "A.X", 1: "A.Y"})

            pairs = pd.concat([Y_C_, Z_C_], axis=1)
            pairs["distance"] = dist_df.iloc[row_indices, col_indices].values.diagonal()

            filtered_BprimeID = pairs["BprimeID"]

            if filtered_BprimeID.empty:
                break

            dist_df = pd.DataFrame(distances)
            dist_df = dist_df.iloc[:, DF2.drop(filtered_BprimeID.tolist()).index.to_numpy()]
            min_row_indices = dist_df.idxmin()
            if verbose is True:
                print(f"min_row_indices", min_row_indices.shape)
            df2 = DF2.iloc[DF2.drop(filtered_BprimeID.tolist()).index.to_numpy(), :].rename(columns={
                0: "x",
                1: "y",
            })
            df3 = DF3.iloc[min_row_indices, :].rename(columns={
                0: "x",
                1: "y",
            })

            def loss_function(params, df2, df3):
                tx, ty, theta = params
                translation_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
                rotation_matrix = np.array(
                    [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
                transform_matrix = translation_matrix.dot(rotation_matrix)
                ones = np.ones((df2.shape[0], 1))
                df2_homogeneous = np.hstack([df2, ones])
                df2_transformed = df2_homogeneous.dot(transform_matrix.T)
                total_distance = 0
                for point, point1 in zip(df2_transformed[:, :2], df2):
                    distances = np.linalg.norm(df3 - point, axis=1)
                    distances2 = np.linalg.norm(point - point1)
                    total_distance += np.sum(distances) + distances2 * 0
                return total_distance

            initial_params = [0, 0, 0]
            result = minimize(loss_function, initial_params, args=(df2.values, df3.values), method='L-BFGS-B')

            optimal_params = result.x
            tx_opt, ty_opt, theta_opt = optimal_params
            translation_matrix_opt = np.array([[1, 0, tx_opt], [0, 1, ty_opt], [0, 0, 1]])
            rotation_matrix_opt = np.array(
                [[np.cos(theta_opt), -np.sin(theta_opt), 0], [np.sin(theta_opt), np.cos(theta_opt), 0], [0, 0, 1]])
            transform_matrix_opt = translation_matrix_opt.dot(rotation_matrix_opt)
            if verbose is True:
                print("Optimal Translation and Rotation Parameters:")
                print("Translation: (", tx_opt, ",", ty_opt, ")")
                print("Rotation (radians):", theta_opt)
                print("Transform Matrix:\n", transform_matrix_opt)

                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.scatter(df2['x'], df2['y'], label='df2 (original)', alpha=0.6)
                plt.scatter(df3['x'], df3['y'], label='df3', alpha=0.6)
                plt.title('Original Data')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.legend()

                plt.subplot(1, 2, 2)
                plt.scatter(df2_transformed[:, 0], df2_transformed[:, 1], label='df2 (transformed)', alpha=0.6)
                plt.scatter(df3['x'], df3['y'], label='df3', alpha=0.6)
                plt.title('Transformed Data')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.legend()

                plt.tight_layout()
                plt.show()

            ones = np.ones((df2.shape[0], 1))
            df2_homogeneous = np.hstack([df2.values, ones])
            df2_transformed = df2_homogeneous.dot(transform_matrix_opt.T)
            ones = np.ones((DF2.shape[0], 1))
            df2_homogeneous = np.hstack([DF2.values, ones])
            DF2 = df2_homogeneous.dot(transform_matrix_opt.T)
            DF2 = pd.DataFrame(DF2)
            DF2 = DF2.iloc[:, 0:2]

            if np.allclose(transform_matrix_opt, np.eye(3), atol=1e-3):
                break

        return DF2
    b_prime_D = perform_transformations(DF2, DF3, Y_c, Z_c, data2_leiden_str, data1_leiden_str)
    if work_dir is not None:
        plt.scatter(a.D[:,0], a.D[:,1], color='red', label='A Points')
        plt.scatter(b_prime_D.values[0], b_prime_D.values[1], color='blue',
                label='B prime transfer Points')
        plt.legend()
        plt.savefig(os.path.join(work_dir, 'further_align.png'))

    a = Data(X=a.X, D=a.D, Label=a.Label)
    b_prime= Data(X=b_prime.X, D=b_prime_D.values, Label=b_prime.Label)
    return b_prime

def find_anchor_points(
        a: Data, # A (n, X_1, D),
        b_prime: Data, # B' (m, X_2, D),
        dist_min: float = 100, # the minimum distance for further alignment
        simpson_index_threshold: float = 0.5, # the simpson index threshold for further alignment
) -> Data: # [(a_i, b_j, distance_i_j)] where a_i is the anchor point in A and b_j is the anchor point in B'
    """
    Find the anchor points for further alignment
    :param a: Data, the first dataset
    :param b_prime: Data, the second dataset
    :return: [(a_i, b_j)] where a_i is the anchor point in A and b_j is the anchor point in B'
    """

    distances = cdist(a.D, b_prime.D, metric='euclidean')
    # distance's shape?
    # [n, m]
    indices_row, indices_col = np.where(distances < dist_min)
    indices = np.vstack([indices_row, indices_col]).T

    # indice example
    # [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)]

    # get the indice_group_by_a
    # {0: [0, 1, 2], 1: [1, 2]}
    indice_group_by_a = {}
    for i, j in indices:
        if i not in indice_group_by_a:
            indice_group_by_a[i] = []
        indice_group_by_a[i].append(j)

    # get the label_group_by_a
    # {0: [a, a, b], 1: [a, b, b]}
    label_group_by_a = {i: a.Label[indice_group_by_a[i]] for i in indice_group_by_a}
    # find the most common label for each group
    # {0: a, 1: b}
    most_common_label_group_by_a = {}
    for i in indice_group_by_a:
        most_common_label_group_by_a[i] = Counter(b_prime.Label[indice_group_by_a[i]]).most_common(1)[0][0]
    
    # calculate the simpson index for each group
    simpson_index_group_by_a = {i: calculate_simpson_index(b_prime.Label[indice_group_by_a[i]]) for i in indice_group_by_a}

    simpson_index = [i for i in simpson_index_group_by_a.keys()]
    for i in simpson_index:
        if simpson_index_group_by_a[i] > simpson_index_threshold:
            del simpson_index_group_by_a[i]

    simpson_index = [i for i in simpson_index_group_by_a.keys()]
    # example for simpson_index
    # [0, 1, 3, 4]

    # get the anchor_points
    # [(0, 0), (1, 1), (3, 2), (4, 3)]
    anchor_points = [(i, j) for i in simpson_index for j in indice_group_by_a[i]]
    anchor_points_with_distance = [(i, j, distances[i, j]) for i, j in anchor_points]
    print("anchor_points_pairs",len(anchor_points_with_distance))

    return anchor_points_with_distance

def find_high_correlation_features(
        a: Data, # A (n, X_1, D),
        b_prime: Data, # B' (m, X_2, D),
        low_threshold: float = 0, # the threshold for low correlation
        high_threshold: float = 0.95, # the threshold for high correlation
        n_features: int = 100, # the number of features to select
        dist_min: float = 100, # the minimum distance for further alignment
) -> Data: # [(a_i, b_j, correlation_i_j)] where a_i is the feature in A and b_j is the feature in B'
    a_X = a.X
    b_prime_X = b_prime.X
    anchor_points_with_distance = find_anchor_points(a, b_prime, dist_min=dist_min)
    # anchor_points_with_distance example
    # [(0, 0, 0.1), (1, 1, 0.2), (3, 2, 0.3), (4, 3, 0.4)]

    anchor_points_indices_a = [i for i, _, _ in anchor_points_with_distance]
    anchor_points_indices_b = [j for _, j, _ in anchor_points_with_distance]
    anchor_points_a_X = a_X[anchor_points_indices_a] # shape: [n', X_1]
    anchor_points_b_X = b_prime_X[anchor_points_indices_b] # shape: [n', X_2]

    anchor_points_a_b_X = np.hstack([anchor_points_a_X, anchor_points_b_X]) # shape: [n', X_1 + X_2]

    # calculate the correlation matrix
    correlation_matrix = np.corrcoef(anchor_points_a_b_X, rowvar=False) # shape: [X_1 + X_2, X_1 + X_2]
    # only keep [X1, X2] (the top right corner)
    correlation_matrix = correlation_matrix[:a_X.shape[1], a_X.shape[1]:]

    # convert to turple (i, j, correlation_i_j)
    correlation_matrix_with_indices = [(i, j, correlation_matrix[i, j]) for i in range(correlation_matrix.shape[0]) for j in range(correlation_matrix.shape[1])]
    # sort by correlation_i_j
    correlation_matrix_with_indices = sorted(correlation_matrix_with_indices, key=lambda x: x[2], reverse=True)

    # select the top n_features
    top_n_features = []
    for i, j, correlation in correlation_matrix_with_indices:
        if len(top_n_features) == n_features:
            break
        if correlation > low_threshold and correlation <= high_threshold and j not in [j for _, j, _ in top_n_features]:
            top_n_features.append((i, j, correlation))    
    print("lenth of high_correlated_features_pairs",len(top_n_features))
    return top_n_features

def cca_featurize(
        a: Data, # A (X_1, D),
        b_prime: Data, # B' (X_2, D),
        correlation_feature_pairs: Data, # [(a_i, b_j, correlation_i_j)] where a_i is the feature in A and b_j is the feature in B'
        n_components: int = 1, # the number of components to keep
        work_dir: str = None, # the working directory, will be created if not exists. Will be used to save the intermediate results.
        verbose: bool = False
) -> Union[np.ndarray, np.ndarray]: # the featurized data for a and b
    a_D = a.D
    b_prime_D = b_prime.D
    distances = cdist(a_D, b_prime_D, metric='euclidean') # shape: [n, m]
    min_row_indices = np.argmin(distances, axis=0) # shape: [n, m]
    feature_a = a.X[:, [i for i, _, _ in correlation_feature_pairs]]
    feature_a2 = a.X[:, [i for i, _, _ in correlation_feature_pairs]]
    feature_a2 = feature_a2[min_row_indices,:]
    feature_b = b_prime.X[:, [j for _, j, _ in correlation_feature_pairs]]
    if verbose is True:
        print(feature_b)
        print(feature_a)
        print(feature_a2)
    cca = CCA(n_components=n_components)
    feature_a2, feature_b = cca.fit_transform(feature_a2, feature_b)
    feature_a = cca.transform(feature_a)
    return feature_a, feature_b

def direct_alignment(
        a: Data, # a (n, X_1, D)
        b_prime: Data, # b' (m, X_2, D)
        work_dir: str = None,
        enable_center_and_scale: bool = False,
) -> Data: # [b（n, [X_2], D）,]
    """
    Direct alignment
    :param a: Data, the first dataset
    :param b_prime: Data, the second dataset
    :param work_dir: str, the working directory, will be created if not exists. Will be used to save the intermediate results.
    :return: Data, the aligned dataset for a
    """
    a_D = a.D
    b_prime_D = b_prime.D
    # center and scale the D for a and b_prime
    if enable_center_and_scale:
        a_D = center_and_scale(a.D)
        b_prime_D = center_and_scale(b_prime.D)

    # Calculate pairwise distances
    # the shape of distances is [n, m] where n is the number of data points in data1 and m is the number of data points in data2
    distances = cdist(a_D, b_prime_D, metric='euclidean') # shape: [n, m]

    # Find the closest point in data2 for each point in data1
    # the shape of min_row_indices is [n, 1] where n is the number of data points in data1
    min_row_indices = np.argmin(distances, axis=0) # shape: [m, 1]

    # calculate the alignment result from data1 to data2
    # the alignment result is a [m, 1] array where m is the number of data points in data1
    # and the value is the index of the closest point in data2

    alignment_a_prime = None if b_prime.X is None else a.X[min_row_indices]
    b_predict = Data(X=alignment_a_prime, D=a_D[min_row_indices], Label=a.Label[min_row_indices])

    if work_dir is not None:
        # plot the direct alignment result
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')

        # plot the data points in data1 with zs = 1 and
        ax.scatter(a.D[:, 0], a.D[:, 1], zs = 1, c=a.Label.astype('int'), marker='o', cmap='tab20')

        # plot the data points in data2 with zs = -1 and color blue
        ax.scatter(b_prime.D[:, 0], b_prime.D[:, 1], zs = -1, c=b_prime.Label.astype('int'), marker='s', cmap='tab20')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend()

        # save the figure to work_dir/direct_alignment.png
        fig.savefig(os.path.join(work_dir, 'direct_alignment.png'))

    return b_predict


def icp_3d_alignment(
        a: Data, # A (n, X_1, D),
        b_prime: Data, # B' (m, X_2, D),
        work_dir: str = None, # the working directory, will be created if not exists. Will be used to save the intermediate results.
        max_iterations: int = 500, # the maximum number of iterations
        tolerance: float = 1e-5, # the tolerance for convergence
        n_components: int = 1, # the number of components to keep
        verbose: bool = False,
        low_threshold: float = 0, # the threshold for low correlation
        high_threshold: float = 0.95, # the threshold for high correlation
        n_features: int = 100, # the number of features to select
        dist_min: float = 100, # the minimum distance for further alignment
    ) -> Tuple[Data, Data]: # (A(n, X_1, D), B_predict (n, X_2, D))

    correlation_feature_pairs = find_high_correlation_features(a, b_prime, low_threshold=low_threshold,
                                                               high_threshold=high_threshold,dist_min = dist_min,n_features=n_features )
    (cca_a, cca_b_prime) = cca_featurize(a, b_prime, correlation_feature_pairs, n_components, work_dir)
    a_d = a.D
    b_prime_d = b_prime.D

    feature_a = np.hstack([a_d, cca_a])
    feature_b_prime = np.hstack([b_prime_d, cca_b_prime])

    # 初始变换矩阵
    transformation_matrix = np.eye(4)  # 初始变换矩阵为4x4单位矩阵

    # 设置迭代参数
    max_iterations = max_iterations
    tolerance = tolerance

    if work_dir is not None:
        # 绘制初始点云
        fig = plt.figure(figsize=(10, 5))

        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(feature_a[:, 0], feature_a[:, 1], feature_a[:, 2], color='blue', label='DY2')
        ax1.scatter(feature_b_prime[:, 0], feature_b_prime[:, 1], feature_b_prime[:, 2], color='red', label='DY1')
        ax1.set_title('Initial Point Clouds')
        ax1.legend()

        # save the figure to work_dir/icp_3d_initial.png
        plt.savefig(os.path.join(work_dir, 'icp_3d_initial.png'))

    iteration = 0
    converged = False
    while not converged and iteration < max_iterations:
        # Step 1: Find correspondences based on spatial proximity
        tree = cKDTree(feature_a)
        distances, indices = tree.query(feature_b_prime)

        corresponding_target_points = feature_a[indices]

        # Step 2: Calculate centroids
        centroid_b_prime = np.mean(feature_b_prime, axis=0, keepdims=True) # shape: [1, 3]
        centroid_a = np.mean(corresponding_target_points, axis=0, keepdims=True) # shape: [1, 3]

        # Step 3: Center the points
        DY1_centered = feature_b_prime - centroid_b_prime # shape: [m, 3]
        DY2_centered = corresponding_target_points - centroid_a # shape: [m, 3]

        # Step 4: Compute the covariance matrix
        H = DY1_centered.T @ DY2_centered # shape: [3, 3]
        # Step 5: Compute the Singular Value Decomposition (SVD)
        U, S, Vt = np.linalg.svd(H) # U: [3, 3], S: [3], Vt: [3, 3]
        R = Vt.T @ U.T # shape: [3, 3]

        # Step 6: Ensure a proper rotation matrix (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T

        # Step 7: Compute the translation vector
        t = centroid_a - centroid_b_prime @ R # shape: [1, 3]

        # Step 8: Form the transformation matrix
        transformation = np.eye(4) # shape: [4, 4]
        transformation[:3, :3] = R # shape: [3, 3]
        transformation[:3, 3] = t.flatten() # shape: [3]

        # Step 9: Apply the transformation
        DY1_homogeneous = np.hstack((feature_b_prime, np.ones((feature_b_prime.shape[0], 1)))) # shape: [m, 4]
        DY1_transformed_homogeneous = DY1_homogeneous @ transformation.T # shape: [m, 4]
        DY1_transformed = DY1_transformed_homogeneous[:, :3] # shape: [m, 3]

        # Step 10: Update DY1|
        feature_b_prime = DY1_transformed # shape: [m, 3]

        # Step 11: Check convergence
        if np.linalg.norm(transformation[:3, 3]) < tolerance:
            converged = True

        iteration += 1

    if work_dir is not None:
        # 绘制最终点云
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(feature_b_prime[:, 0], feature_b_prime[:, 1], feature_b_prime[:, 2], color='red', label='DY1 (transformed)')
        ax2.scatter(feature_a[:, 0], feature_a[:, 1], feature_a[:, 2], color='blue', label='DY2')
        ax2.set_title('Final Point Clouds')
        ax2.legend()
        plt.tight_layout()
        # save the figure to work_dir/icp_3d.png
        plt.savefig(os.path.join(work_dir, 'icp_3d.png'))
    if verbose is True:
        print("Final transformation matrix:")
        print(transformation)

    a = Data(X=a.X, D=feature_a, Label=a.Label)
    b_prime = Data(X=b_prime.X, D=feature_b_prime, Label=b_prime.Label)

    # run direct alignment with the DY1 and DY2
    return a, b_prime
    
def icp_2d_alignment(
        a: Data, # A (X_1, D),
        b_prime: Data, # B' (X_2, D),
        work_dir: str = None, # the working directory, will be created if not exists. Will be used to save the intermediate results.
        max_iterations: int = 500, # the maximum number of iterations
        tolerance: float = 1e-5, # the tolerance for convergence
        verbose: bool = False
    ) -> Data: # A' (X_2, D)

    a_D = a.D  # Example DYGW1 features
    b_prime_D = b_prime.D  # Example DYGW2 features
    transformation = np.eye(2)  # 初始变换矩阵为2x2的单位矩阵

    if work_dir is not None:
        # plot the initial point clouds and save the figure to work_dir/icp_2d_initial.png
        # 创建一个新的图形窗口
        plt.figure(figsize=(10, 5))

        # 绘制初始的DY1和DY2点云
        plt.subplot(1, 2, 1)
        plt.title('Initial Point Clouds')
        dy1_color = 'red'
        dy2_color = 'blue'
        plt.scatter(a_D[:, 0], a_D[:, 1], color=dy1_color, label='DY1')
        plt.scatter(b_prime_D[:, 0], b_prime_D[:, 1], color=dy2_color, label='DY2')
        plt.legend()

        plt.savefig(os.path.join(work_dir, 'icp_2d_initial.png'))

    iteration = 0
    converged = False

    while not converged and iteration < max_iterations:
        # Step 1: Find correspondences based on spatial proximity
        tree = cKDTree(a_D)
        distances, indices = tree.query(b_prime_D)

        corresponding_target_points = a_D[indices]

        # Step 2: Optimize transformation using spatial and feature distances
        # Example: simple averaging transformation for illustration
        transformation = np.mean(corresponding_target_points - b_prime_D, axis=0, keepdims=True)

        # Step 3: Apply transformation to b_prime_D
        b_prime_D = b_prime_D + transformation

        # Step 4: Check convergence
        if np.linalg.norm(transformation) < tolerance:
            print(f"Converged after {iteration} iterations.")
            converged = True

        iteration += 1
    # save the figure to work_dir/icp_2d.png
    if work_dir is not None:
        # 绘制最终的DY1和DY2点云
        plt.subplot(1, 2, 2)
        plt.title('Final Point Clouds')
        plt.scatter(a_D[:, 0], a_D[:, 1], color=dy1_color, label='DY1')
        plt.scatter(b_prime_D[:, 0], b_prime_D[:, 1], color=dy2_color, label='DY2 (transformed)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(work_dir, 'icp_2d.png'))
    if verbose is True:
        print("Final transformation matrix:")
        print(transformation)

    a = Data(X=a.X, D=a_D, Label=a.Label)
    b_prime = Data(X=b_prime.X, D=b_prime_D, Label=b_prime.Label)

    # run direct alignment with the DY1 and DY2
    return b_prime

def fgw_2d_alignment(
        a: Data, # A (X_1, D),
        b_prime: Data, # B' (X_2, D),
        work_dir: str = None, # the working directory, will be created if not exists. Will be used to save the intermediate results.
        alpha: float = 0.5, # the balance parameter
        max_iter: int = 2000, # the maximum number of iterations
        tol_rel: float = 1e-9, # the relative tolerance
        tol_abs: float = 1e-9, # the absolute tolerance
        armijo: bool = True, # whether to use Armijo line search
) -> pd.DataFrame: # the aligned pairs of point pairs (data1.X, data1.Y, data2.X, data2.Y)
    
    # 假设 P 和 Q 是两个点云，形状分别为 (n, d) 和 (m, d)
    Q = a.D  # Example DYGW1 features
    P = b_prime.D  # Example DYGW2 features
    
    # 特征矩阵（这里假设特征与坐标相同）
    F_P = P
    F_Q = Q
    
    # 数据标准化
    scaler = StandardScaler()
    P = scaler.fit_transform(P)
    Q = scaler.fit_transform(Q)

    # 数据归一化
    scaler = MinMaxScaler()
    P = scaler.fit_transform(P)
    Q = scaler.fit_transform(Q)

    # 计算几何距离矩阵
    C_P = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    C_Q = np.linalg.norm(Q[:, None, :] - Q[None, :, :], axis=2)

    # 计算特征距离矩阵
    M = np.linalg.norm(F_P[:, None, :] - F_Q[None, :, :], axis=2)

    # 初始化权重，均匀分布
    init_p = np.ones((P.shape[0],)) / P.shape[0]
    init_q = np.ones((Q.shape[0],)) / Q.shape[0]

    # 设置参数
    params = {
        'max_iter': max_iter,    # 增加最大迭代次数
        'tol_rel': tol_rel,     # 调整相对容差
        'tol_abs': tol_abs,     # 调整绝对容差
        'armijo': armijo       # 使用Armijo线搜索
    }

    # 计算FGW最优传输计划
    pi, log = ot.gromov.fused_gromov_wasserstein(M,C_P, C_Q, init_p, init_q, alpha=alpha, log=True, **params)

    # 对齐点云 P 到 Q
    P_aligned = np.dot(pi, Q)
    scaler = StandardScaler()
    P_aligned = scaler.fit_transform(P_aligned)
    scaler = MinMaxScaler()
    P_aligned = scaler.fit_transform(P_aligned)

    P_aligned = pd.DataFrame(P_aligned)
    scaledata = center_and_scale(P_aligned)
    P_aligned = pd.DataFrame(scaledata, columns=P_aligned.columns).to_numpy()

    Q = pd.DataFrame(Q)
    scaledata = center_and_scale(Q)
    Q = pd.DataFrame(scaledata, columns=Q.columns).to_numpy()

    # 保存可视化结果
    if work_dir is not None:

        # 可视化
        plt.figure(figsize=(18, 6))

        # 原始点云
        plt.subplot(1, 3, 1)
        plt.scatter(P[:, 0], P[:, 1], color='red', label='Bprime', alpha=0.6)
        plt.scatter(Q[:, 0], Q[:, 1], color='blue', label='A', alpha=0.6)
        plt.title('Original Point Clouds')
        plt.legend()

        # 对齐后的点云
        plt.subplot(1, 3, 2)
        plt.scatter(P_aligned[:, 0], P_aligned[:, 1], color='red', label='Aligned Bprime', alpha=0.6)
        plt.scatter(Q[:, 0], Q[:, 1], color='blue', label='A', alpha=0.6)
        plt.title('Aligned Point Clouds')
        plt.legend()

        # 可视化传输计划
        plt.subplot(1, 3, 3)
        plt.imshow(pi, cmap='hot', interpolation='nearest')
        plt.title('Transport Plan')
        plt.colorbar()
        plt.savefig(os.path.join(work_dir, 'fgw_2d.png'))

    b_prime = Data(X=b_prime.X, D=P_aligned, Label=b_prime.Label)

    # direct alignment
    return b_prime

def fgw_3d_alignment(
        a: Data, # A (X_1, D),
        b_prime: Data, # B' (X_2, D),
        work_dir: str = None, # the working directory, will be created if not exists. Will be used to save the intermediate results.
        alpha: float = 0.5, # the balance parameter
        max_iter: int = 2000, # the maximum number of iterations
        tol_rel: float = 1e-9, # the relative tolerance
        tol_abs: float = 1e-9, # the absolute tolerance
        armijo: bool = True, # whether to use Armijo line search
        n_components: int = 1, # the number of components to keep
        low_threshold: float = 0, # the threshold for low correlation
        verbose: bool = False
) -> Tuple[Data, Data]: # (A(n, X_1, D), B_predict (n, X_2, D))
    correlation_feature_pairs = find_high_correlation_features(a, b_prime, low_threshold=low_threshold)
    (cca_a, cca_b_prime) = cca_featurize(a, b_prime, correlation_feature_pairs, n_components, work_dir)
    Q = a.D
    P = b_prime.D
    # 特征矩阵（这里假设特征与坐标相同）
    F_Q = np.hstack([cca_a])
    F_P = np.hstack([cca_b_prime])
    
    # 数据标准化
    scaler = StandardScaler()
    P = scaler.fit_transform(P)
    Q = scaler.fit_transform(Q)
    F_Q = scaler.fit_transform(F_Q)
    F_P = scaler.fit_transform(F_P)
    # 数据归一化
    scaler = MinMaxScaler()
    P = scaler.fit_transform(P)
    Q = scaler.fit_transform(Q)
    F_Q = scaler.fit_transform(F_Q)
    F_P = scaler.fit_transform(F_P)

    # 计算几何距离矩阵
    C_P = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    C_Q = np.linalg.norm(Q[:, None, :] - Q[None, :, :], axis=2)
    
    # 计算特征距离矩阵
    M = np.linalg.norm(F_P[:, None, :] - F_Q[None, :, :], axis=2)
    
    # 初始化权重，均匀分布
    init_p = np.ones((P.shape[0],)) / P.shape[0]
    init_q = np.ones((Q.shape[0],)) / Q.shape[0]
    
    # 设置参数
    params = {
        'max_iter': max_iter,    # 增加最大迭代次数
        'tol_rel': tol_rel,     # 调整相对容差
        'tol_abs': tol_abs,     # 调整绝对容差
        'armijo': armijo       # 使用Armijo线搜索
    }
    
    # 计算FGW最优传输计划
    pi, log = ot.gromov.fused_gromov_wasserstein(M,C_P, C_Q, init_p, init_q, alpha=alpha, log=True, **params)
    
    # 对齐点云 P 到 Q
    P_aligned = np.dot(pi, np.hstack((Q,F_Q)))
    scaler = StandardScaler()
    P_aligned = scaler.fit_transform(P_aligned)
    scaler = MinMaxScaler()
    P_aligned = scaler.fit_transform(P_aligned)
    
    # 保存可视化结果
    if work_dir is not None:
        
        # 可视化
        plt.figure(figsize=(18, 6))

        # 原始点云
        plt.subplot(1, 3, 1)
        plt.scatter(P[:, 0], P[:, 1], color='red', label='P', alpha=0.6)
        plt.scatter(Q[:, 0], Q[:, 1], color='blue', label='Q', alpha=0.6)
        plt.title('Original Point Clouds')
        plt.legend()

        # 对齐后的点云
        plt.subplot(1, 3, 2)
        plt.scatter(P_aligned[:, 0], P_aligned[:, 1], color='red', label='Aligned P', alpha=0.6)
        plt.scatter(Q[:, 0], Q[:, 1], color='blue', label='Q', alpha=0.6)
        plt.title('Aligned Point Clouds')
        plt.legend()

        # 可视化传输计划
        plt.subplot(1, 3, 3)
        plt.imshow(pi, cmap='hot', interpolation='nearest')
        plt.title('Transport Plan')
        plt.colorbar()
        plt.savefig(os.path.join(work_dir, 'fgw_2d.png'))

    a = Data(X=a.X, D=np.hstack((Q,F_Q)), Label=a.Label)
    b_prime = Data(X=b_prime.X, D=P_aligned, Label=b_prime.Label)

    # direct alignment
    return a, b_prime

def convert_to_array(x):
    if isinstance(x, csr_matrix):
        return x.toarray()
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise ValueError("Unsupported input type. Must be csr_matrix or ndarray.")


def plot_b_predict(
        b_prime: Data, # B' (X_2, D)
        work_dir: str = None, # the working directory, will be created if not exists. Will be used to save the intermediate results.
        save: bool = True
):
    plt.close()
    b_prime_D = b_prime.D
    b_prime_labels = b_prime.Label.astype(int)
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
