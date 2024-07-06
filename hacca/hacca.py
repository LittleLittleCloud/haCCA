import os, sys
from typing import Union
from anndata import AnnData
from matplotlib import pyplot as plt
import scanpy as sc
from sklearn.cross_decomposition import CCA
from sklearn.metrics import adjusted_rand_score
from sklearn.manifold import TSNE
from scipy.sparse import csr_matrix
from scipy.sparse import csr_matrix
import numpy as np
import umap.umap_ as umap
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from scipy.spatial.distance import cdist
import random
from scipy.spatial import cKDTree
import ot
from collections import Counter
from hacca.data import Data

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

def calculate_accuracy_for_pairwise_alignment(
        pairs: pd.DataFrame, # the aligned pairs of point pairs (data1.X, data1.Y, data2.X, data2.Y)
) -> float:
    same_values_mask_directmerge = pairs['data1_cluster'] == pairs['data2_cluster']
    accuracy_directmerge = sum(same_values_mask_directmerge) / len(same_values_mask_directmerge)
    print(f"Accuracy for pairwise alignment: {accuracy_directmerge}")
    return accuracy_directmerge

def label_transfer_ari(source: Data, target: Data):
    """
    Compute the Adjusted Rand Index (ARI) between the labels of the source and target datasets.
    
    Parameters:
    - source: Data instance, the source dataset with original labels.
    - target: Data instance, the target dataset with transferred labels.
    
    Returns:
    - ARI: The Adjusted Rand Index, measuring the similarity between the source and target labels.
    """
    # Assuming source.Label and target.Label are numpy arrays of labels
    ari_score = adjusted_rand_score(source.Label, target.Label)
    return ari_score

def loss(predict: Data, target: Data, alpha: float = 0.5) -> float:
    """
    Calculate the loss between the predict and target data
    :param predict: Data, the predict data
    :param target: Data, the target data
    :param alpha: float, the balance parameter
    :return: float, the loss
    """
    # calculate the L2 loss between the predict and target D matrix
    loss_D = np.linalg.norm(predict.X - target.X, ord=2)

    # calculate the label match accuracy
    same_values_mask = predict.Label == target.Label
    accuracy = sum(same_values_mask) / len(same_values_mask)
    ari = label_transfer_ari(predict, target)
    
    return loss_D, accuracy, ari

def further_alignment(
        a: Data, # A (X_1, D),
        b_prime: Data, # B' (X_2, D),
        work_dir: str = None, # the working directory, will be created if not exists. Will be used to save the intermediate results.
        dist_min: float = 2, # the minimum distance for further alignment
) -> Data: # b (n, [X_2], D)
    """
    Further alignment
    :param a: Data, the first dataset
    :param b_prime: Data, the second dataset
    :param work_dir: str, the working directory, will be created if not exists. Will be used to save the intermediate results.
    :return: Data, the aligned dataset for a
    """

    # further alignment
    pass

    return direct_alignment(a, b_prime, work_dir, enable_center_and_scale=False)

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

    return anchor_points_with_distance

def find_high_correlation_features(
        a: Data, # A (n, X_1, D),
        b_prime: Data, # B' (m, X_2, D),
        low_threshold: float = 0.5, # the threshold for high correlation
        high_threshold: float = 0.95, # the threshold for low correlation
        n_features: int = 100, # the number of features to select
) -> Data: # [(a_i, b_j, correlation_i_j)] where a_i is the feature in A and b_j is the feature in B'
    a_X = a.X
    b_prime_X = b_prime.X
    anchor_points_with_distance = find_anchor_points(a, b_prime)
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

    return top_n_features

def cca_featurize(
        a: Data, # A (X_1, D),
        b_prime: Data, # B' (X_2, D),
        correlation_feature_pairs: Data, # [(a_i, b_j, correlation_i_j)] where a_i is the feature in A and b_j is the feature in B'
        n_components: int = 1, # the number of components to keep
        work_dir: str = None, # the working directory, will be created if not exists. Will be used to save the intermediate results.
) -> Union[np.ndarray, np.ndarray]: # the featurized data for a and b
    feature_a = a.X[:, [i for i, _, _ in correlation_feature_pairs]]
    feature_b = b_prime.X[:, [j for _, j, _ in correlation_feature_pairs]]
    cca = CCA(n_components=n_components)
    feature_a, feature_b = cca.fit_transform(feature_a, feature_b)

    return feature_a, feature_b

def direct_alignment(
        a: Data, # a (n, X_1, D)
        b_prime: Data, # b' (m, X_2, D)
        work_dir: str = None,
        enable_center_and_scale: bool = True,
) -> Data: # b（n, [X_2], D）
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
    distances = cdist(a_D, b_prime_D, metric='euclidean')

    # Find the closest point in data2 for each point in data1
    # the shape of min_row_indices is [n, 1] where n is the number of data points in data1
    min_row_indices = np.argmin(distances, axis=1)

    # calculate the alignment result from data1 to data2
    # the alignment result is a [n, 1] array where n is the number of data points in data1
    # and the value is the index of the closest point in data2

    alignment_a_prime = None if b_prime.X is None else b_prime.X[min_row_indices]
    a_prime = Data(X=alignment_a_prime, D=b_prime_D[min_row_indices], Label=b_prime.Label[min_row_indices])

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

    return a_prime

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

def icp_3d_alignment(
        a: Data, # A (n, X_1, D),
        b_prime: Data, # B' (m, X_2, D),
        work_dir: str = None, # the working directory, will be created if not exists. Will be used to save the intermediate results.
        max_iterations: int = 500, # the maximum number of iterations
        tolerance: float = 1e-5, # the tolerance for convergence
        n_components: int = 1, # the number of components to keep
    ) -> Data: # A' (n, X_2, D)

    correlation_feature_pairs = find_high_correlation_features(a, b_prime)
    (cca_a, cca_b_prime) = cca_featurize(a, b_prime, correlation_feature_pairs, n_components, work_dir)
    a_d = a.D
    b_prime_d = b_prime.D

    feature_a = np.hstack([a_d, cca_a])
    feature_b_prime = np.hstack([b_prime_d, cca_b_prime])

    # 初始变换矩阵
    transformation_matrix = np.eye(4)  # 初始变换矩阵为4x4单位矩阵

    # 设置迭代参数
    max_iterations = 500
    tolerance = 1e-5

    if work_dir is not None:
        # 绘制初始点云
        fig = plt.figure(figsize=(10, 5))

        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(feature_a[:, 0], feature_a[:, 1], feature_a[:, 2], color='blue', label='DY1')
        ax1.scatter(feature_b_prime[:, 0], feature_b_prime[:, 1], feature_b_prime[:, 2], color='red', label='DY2')
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
        ax2.scatter(feature_b_prime[:, 0], feature_b_prime[:, 1], feature_b_prime[:, 2], color='blue', label='DY1 (transformed)')
        ax2.scatter(feature_a[:, 0], feature_a[:, 1], feature_a[:, 2], color='red', label='DY2')
        ax2.set_title('Final Point Clouds')
        ax2.legend()
        plt.tight_layout()
        print("Final transformation matrix:")
        print(transformation)

        # save the figure to work_dir/icp_3d.png
        plt.savefig(os.path.join(work_dir, 'icp_3d.png'))

    a = Data(X=a.X, D=feature_a, Label=a.Label)
    b_prime = Data(X=b_prime.X, D=feature_b_prime, Label=b_prime.Label)

    # run direct alignment with the DY1 and DY2
    return direct_alignment(a, b_prime, work_dir, enable_center_and_scale=False)
    
def icp_2d_alignment(
        a: Data, # A (X_1, D),
        b_prime: Data, # B' (X_2, D),
        work_dir: str = None, # the working directory, will be created if not exists. Will be used to save the intermediate results.
        max_iterations: int = 500, # the maximum number of iterations
        tolerance: float = 1e-5, # the tolerance for convergence
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
    print("Final transformation matrix:")
    print(transformation)

    a = Data(X=a.X, D=a_D, Label=a.Label)
    b_prime = Data(X=b_prime.X, D=b_prime_D, Label=b_prime.Label)

    # run direct alignment with the DY1 and DY2
    return direct_alignment(a, b_prime, work_dir, enable_center_and_scale=False)

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
    P = a.D  # Example DYGW1 features
    Q = b_prime.D  # Example DYGW2 features
    
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

    a = Data(X=a.X, D=P_aligned, Label=a.Label)
    b_prime = Data(X=b_prime.X, D=Q, Label=b_prime.Label)

    # direct alignment
    return direct_alignment(a, b_prime, work_dir, enable_center_and_scale=False)

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
) -> pd.DataFrame: # the aligned pairs of point pairs (data1.X, data1.Y, data2.X, data2.Y)
    correlation_feature_pairs = find_high_correlation_features(a, b_prime)
    (cca_a, cca_b_prime) = cca_featurize(a, b_prime, correlation_feature_pairs, n_components, work_dir)
    P = a.D
    Q = b_prime.D
    # 特征矩阵（这里假设特征与坐标相同）
    F_P = np.hstack([cca_a])
    F_Q = np.hstack([cca_b_prime])
    
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

    a = Data(X=a.X, D=P_aligned, Label=a.Label)
    b_prime = Data(X=b_prime.X, D=Q, Label=b_prime.Label)

    # direct alignment
    return direct_alignment(a, b_prime, work_dir, enable_center_and_scale=False)
    

def hacca(
        a: Data, # A (X_1, D)
        b_prime: Data, # B' (X_2, D)
        work_dir: str = None, # the working directory, will be created if not exists. Will be used to save the intermediate results.
    ) -> Data: # A' (X_2, D)
    """
    HACCA
    :param data1: AnnData, the first dataset
    :param data2: AnnData, the second dataset
    :return: B', the aligned and clustered data
    """
    print("sxt shi sha bi")
    # direct alignment
    a_prime = direct_alignment(a, b_prime, data1_spatial_results, data2_spatial_results, data1_leiden_str, data2_leiden_str, work_dir)
    # calculate the accuracy for pairwise alignment of direct alignment
    calculate_accuracy_for_pairwise_alignment(direct_alignement_pair)

def convert_to_array(x):
    if isinstance(x, csr_matrix):
        return x.toarray()
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise ValueError("Unsupported input type. Must be csr_matrix or ndarray.")
    
if __name__ == '__main__':
    # Load data
    import os
    # get the path of current file
    cwd = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(cwd, '..', 'data')
    work_dir = os.path.join(cwd, '..', 'work')
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    a = sc.read_h5ad(os.path.join(data_path, 'A1.h5ad'))
    b_prime = sc.read_h5ad(os.path.join(data_path, 'A2.h5ad'))

    # create a and b_prime
    a = Data(X=a.X.toarray(), D=a.obsm['spatial'], Label=a.obs['leiden'].to_numpy())
    b_prime = Data(X=b_prime.X.toarray(), D=b_prime.obsm['spatial'], Label=b_prime.obs['leiden'].to_numpy())
    b_truth = b_prime
    a_prime_truth = a

    # icp 3d alignment
    icp_3d_work_dir = os.path.join(work_dir, 'icp_3d')
    if not os.path.exists(icp_3d_work_dir):
        os.makedirs(icp_3d_work_dir)
    b_predict = icp_3d_alignment(a, b_prime, icp_3d_work_dir)
    icp_3d_loss = loss(b_predict, b_truth)
    print(f"ICP 3D: loss: {icp_3d_loss}")

    # fgw 3d alignment
    fgw_3d_work_dir = os.path.join(work_dir, 'fgw_3d')
    if not os.path.exists(fgw_3d_work_dir):
        os.makedirs(fgw_3d_work_dir)
    b_predict = fgw_3d_alignment(a, b_prime, fgw_3d_work_dir)
    fgw_3d_loss = loss(b_predict, b_truth)
    print(f"FGW 3D: loss: {fgw_3d_loss}")

    # Run FGW 2D alignment
    fgw_2d_work_dir = os.path.join(work_dir, 'fgw_2d')
    if not os.path.exists(fgw_2d_work_dir):
        os.makedirs(fgw_2d_work_dir)
    b_predict = fgw_2d_alignment(a, b_prime, fgw_2d_work_dir)
    fgw_2d_loss = loss(b_predict, b_truth)
    print(f"FGW 2D: loss: {fgw_2d_loss}")

    # Run ICP 2D alignment
    icp_2d_work_dir = os.path.join(work_dir, 'icp_2d')
    if not os.path.exists(icp_2d_work_dir):
        os.makedirs(icp_2d_work_dir)

    b_predict = icp_2d_alignment(a, b_prime, icp_2d_work_dir)
    icp_2d_loss = loss(b_predict, b_truth)
    print(f"ICP 2D: loss: {icp_2d_loss}")

    # Run direct alignment
    direct_alignment_work_dir = os.path.join(work_dir, 'direct_alignment')
    if not os.path.exists(direct_alignment_work_dir):
        os.makedirs(direct_alignment_work_dir)
    b_predict = direct_alignment(a, b_prime, direct_alignment_work_dir)
    direct_alignment_loss = loss(b_predict, b_truth)
    print(f"Direct alignment w/ center and scale: loss: {direct_alignment_loss}")

    # Run direct alignment without center and scale
    direct_alignment_work_dir = os.path.join(work_dir, 'direct_alignment_no_center_and_scale')
    if not os.path.exists(direct_alignment_work_dir):
        os.makedirs(direct_alignment_work_dir)

    b_predict = direct_alignment(a, b_prime, direct_alignment_work_dir, enable_center_and_scale=False)
    direct_alignment_loss = loss(b_predict, b_truth)
    print(f"Direct alignment w/o center and scale: loss: {direct_alignment_loss}")
    # Run HACCA
    # hacca(a, b, work_dir)