import os, sys
from anndata import AnnData
from matplotlib import pyplot as plt
import scanpy as sc
from sklearn.cross_decomposition import CCA
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

def calculate_accuracy_for_pairwise_alignment(
        pairs: pd.DataFrame, # the aligned pairs of point pairs (data1.X, data1.Y, data2.X, data2.Y)
) -> float:
    same_values_mask_directmerge = pairs['data1_cluster'] == pairs['data2_cluster']
    accuracy_directmerge = sum(same_values_mask_directmerge) / len(same_values_mask_directmerge)
    print(f"Accuracy for pairwise alignment: {accuracy_directmerge}")
    return accuracy_directmerge

def loss(predict: Data, target: Data, alpha: float = 0.5) -> float:
    """
    Calculate the loss between the predict and target data
    :param predict: Data, the predict data
    :param target: Data, the target data
    :param alpha: float, the balance parameter
    :return: float, the loss
    """
    # calculate the L2 loss between the predict and target D matrix
    loss_D = np.linalg.norm(predict.D - target.D, ord=2)

    # calculate the label match accuracy
    same_values_mask = predict.Label == target.Label
    accuracy = sum(same_values_mask) / len(same_values_mask)
    return loss_D, accuracy

def direct_alignment(
        a: Data, # a (X_1, D)
        b_prime: Data, # b' (X_2, D)
        work_dir: str = None,
        enable_center_and_scale: bool = True,
) -> Data: # a'（[X_2], D）
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

# def direct_alignment(
#         adata1: AnnData, # A, e.g.: [1000, 2500]
#         adata2: AnnData, # B', e.g: [1000, 4999]
#         data1: pd.DataFrame, # A, [n, 2], which is the x, y coordinates of the points in the first dataset
#         data2: pd.DataFrame, # B', [n, 2], which is the x, y coordinates of the points in the second dataset
#         data1_leiden_str: np.ndarray, # the leiden clustering result of the first dataset
#         data2_leiden_str: np.ndarray, # the leiden clustering result of the second dataset
#         work_dir: str = None, # the working directory, will be created if not exists. Will be used to save the intermediate results.
#     ) -> pd.DataFrame: # the aligned pairs of point pairs (data1.X, data1.Y, data2.X, data2.Y, data1.cluster, data2.cluster)
#     """
#     Direct alignment
#     :param data1: np.ndarray, the first dataset
#     :param data2: np.ndarray, the second dataset
#     """
#     Y_c = data1
#     Z_c = data2
#     distances = cdist(data1, data2, metric='euclidean')
#     dist_df = pd.DataFrame(distances)
#     min_row_indices = dist_df.idxmin()
#     Y_C_ = Y_c.iloc[:,:]
#     Y_C_['data2_ID'] = Y_C_.index
#     Y_C_ = Y_C_.reset_index()
#     Y_C_["data2_cluster"] = data2_leiden_str.astype(int)
#     Y_C_["data2.Z"] = -1
#     Y_C_ = Y_C_.drop(columns=['index'])
#     Y_C_ = Y_C_.rename(columns={0: "data2.X", 1:"data2.Y"})
#     Z_C_=Z_c.iloc[min_row_indices,]
#     Z_C_['data1_ID'] = Z_C_.index
#     Z_C_=Z_C_.reset_index()
#     Z_C_["data1_cluster"] = data1_leiden_str[min_row_indices].astype(int)
#     Z_C_=Z_C_.drop(columns=['index'])
#     Z_C_["data1.Z"] = 1
#     Z_C_=Z_C_.rename(columns={0: "data1.X", 1:"data1.Y"})
#     pairs = pd.concat([Y_C_,Z_C_],axis=1)
#     #pairs["distance"] = dist_df.iloc[row_indices,col_indices].values.diagonal()  //diagonal要求内存过大

#     # plot the direct alignment result
#     fig = plt.figure(figsize=(10,10))
#     ax = fig.add_subplot(111, projection='3d')
#     plt.scatter(Z_c.iloc[:, 0], Z_c.iloc[:, 1],zs=1, c=data1_leiden_str[:].astype(int), cmap='tab20', s=20, alpha=0.5)
#     plt.scatter(Y_c.iloc[:, 0], Y_c.iloc[:, 1],zs=-1, c=data2_leiden_str.astype(int), cmap='tab20', s=20, alpha=0.5,marker='s')
#     #for i in range(0,pairs.shape[0]):
#     #    plt.plot(pairs.iloc[i,[0,5]], pairs.iloc[i,[1,6]], pairs.iloc[i,[4,9]], 'gray', linewidth=0.2)
#     plt.xlabel('Component 1')
#     plt.ylabel('Component 2')
#     plt.legend()

#     # save the figure to work_dir/direct_alignment.png
#     if work_dir is not None:
#         fig.savefig(os.path.join(work_dir, 'direct_alignment.png'))

#     plt.figure(figsize=(6, 6))
#     plt.scatter(pd.DataFrame(adata2.obsm['spatial']).iloc[:, 0],pd.DataFrame(adata2.obsm['spatial']).iloc[:, 1], 
#                 c=[color_mapping[category] for category in pairs["data1_cluster"].tolist()], s=20, alpha=1)
#     plt.ylabel('Y')
#     unique_categories = pairs["data1_cluster"].unique()
#     handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_mapping[cat], markersize=10) for cat in unique_categories]
    
#     plt.legend(handles, unique_categories, title='Cluster ID', loc='upper right')

#     # save the figure to work_dir/direct_alignment_coloring.png
#     if work_dir is not None:
#         plt.savefig(os.path.join(work_dir, 'direct_alignment_coloring.png'))
    
#     return pairs
        

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

# def icp_2d_alignment(
#         adata1: AnnData, # A, e.g.: [1000, 2500]
#         adata2: AnnData, # B', e.g: [1000, 4999]
#         data1_spatial_results: pd.DataFrame, # the spatial results of the first dataset
#         data2_spatial_results: pd.DataFrame, # the spatial results of the second dataset
#         data1_leiden_str: np.ndarray, # the leiden clustering result of the first dataset
#         data2_leiden_str: np.ndarray, # the leiden clustering result of the second dataset
#         work_dir: str = None, # the working directory, will be created if not exists. Will be used to save the intermediate results.
#         max_iterations: int = 500, # the maximum number of iterations
#         tolerance: float = 1e-5, # the tolerance for convergence
#     ) -> pd.DataFrame: # the aligned pairs of point pairs (data1.X, data1.Y, data2.X, data2.Y)
#     DY1 = data1_spatial_results.T.to_numpy()  # Example DYGW1 features
#     DY2 = data2_spatial_results.T.to_numpy()  # Example DYGW2 features
#     transformation = np.eye(2)  # 初始变换矩阵为2x2的单位矩阵
#     # 创建一个新的图形窗口
#     plt.figure(figsize=(10, 5))

#     # 绘制初始的DY1和DY2点云
#     plt.subplot(1, 2, 1)
#     plt.title('Initial Point Clouds')
#     dy1_color = 'red'
#     dy2_color = 'blue'
#     plt.scatter(DY1[0], DY1[1], color=dy1_color, label='DY1')
#     plt.scatter(DY2[0], DY2[1], color=dy2_color, label='DY2')
#     plt.legend()

#     iteration = 0
#     converged = False

#     while not converged and iteration < max_iterations:
#         # Step 1: Find correspondences based on spatial proximity
#         tree = cKDTree(DY1.T)
#         distances, indices = tree.query(DY2.T)

#         corresponding_target_points = DY1[:, indices]

#         # Step 2: Optimize transformation using spatial and feature distances
#         # Example: simple averaging transformation for illustration
#         transformation = np.mean(corresponding_target_points - DY2, axis=1, keepdims=True)

#         # Step 3: Apply transformation to DY1
#         DY2 = DY2 + transformation

#         # Step 4: Check convergence
#         if np.linalg.norm(transformation) < tolerance:
#             print(f"Converged after {iteration} iterations.")
#             converged = True

#         iteration += 1

#     # 绘制最终的DY1和DY2点云
#     plt.subplot(1, 2, 2)
#     plt.title('Final Point Clouds')
#     plt.scatter(DY1[0], DY1[1], color=dy1_color, label='DY1')
#     plt.scatter(DY2[0], DY2[1], color=dy2_color, label='DY2 (transformed)')
#     plt.legend()
#     plt.tight_layout()
#     # save the figure to work_dir/icp_2d.png
#     if work_dir is not None:
#         plt.savefig(os.path.join(work_dir, 'icp_2d.png'))
#     print("Final transformation matrix:")
#     print(transformation)
#     # convert DY1 and DY2 to pd.DataFrame with the same column of data1_spatial_results and data2_spatial_results
#     DY1 = pd.DataFrame(DY1.T, columns=data1_spatial_results.columns)
#     DY2 = pd.DataFrame(DY2.T, columns=data2_spatial_results.columns)
#     # run direct alignment with the DY1 and DY2
#     return direct_alignment(adata1, adata2, DY1, DY2, data1_leiden_str, data2_leiden_str, work_dir)

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

    # step 1: rough alignment by feature-aided affine transformation
    # the following features are selected manually.
    # data1_pca_results = a.obsm['spatial']  # PCA 结果
    # data1_leiden_str = a.obs["leiden"].to_numpy()

    # data2_pca_results = b_prime.obsm['X_umap']  # PCA 结果
    # data2_leiden_str = b_prime.obs["leiden"].to_numpy()

    # data1_spatial_results = pd.DataFrame(a.obsm['spatial'])
    # scaled_data = center_and_scale(data1_spatial_results)
    # data1_spatial_results = pd.DataFrame(scaled_data, columns=data1_spatial_results.columns)
    # data2_spatial_results = pd.DataFrame(b_prime.obsm['spatial'])
    # scaled_data = center_and_scale(data2_spatial_results)
    # data2_spatial_results = pd.DataFrame(scaled_data, columns=data2_spatial_results.columns)

    # direct alignment
    a_prime = direct_alignment(a, b_prime, data1_spatial_results, data2_spatial_results, data1_leiden_str, data2_leiden_str, work_dir)
    # calculate the accuracy for pairwise alignment of direct alignment
    calculate_accuracy_for_pairwise_alignment(direct_alignement_pair)

    # # direct alignment with icp 2d
    # icp_2d_work_dir = os.path.join(work_dir, 'icp_2d')
    # if not os.path.exists(icp_2d_work_dir):
    #     os.makedirs(icp_2d_work_dir)
    # icp_2d_pair = icp_2d_alignment(a, b_prime, data1_spatial_results, data2_spatial_results, data1_leiden_str, data2_leiden_str, icp_2d_work_dir)
    # # calculate the accuracy for pairwise alignment of icp 2d
    # calculate_accuracy_for_pairwise_alignment(icp_2d_pair)

    # # direct alignment with fgw 2d
    # fgw_2d_work_dir = os.path.join(work_dir, 'fgw_2d')
    # if not os.path.exists(fgw_2d_work_dir):
    #     os.makedirs(fgw_2d_work_dir)
    # fgw_2d_pair = fgw_2d_alignment(a, b_prime, data1_spatial_results, data2_spatial_results, data1_leiden_str, data2_leiden_str, fgw_2d_work_dir)
    # # calculate the accuracy for pairwise alignment of fgw 2d
    # calculate_accuracy_for_pairwise_alignment(fgw_2d_pair)

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
    b = sc.read_h5ad(os.path.join(data_path, 'A2.h5ad'))

    # create a and b_prime
    a = Data(X=None, D=a.obsm['spatial'], Label=a.obs['leiden'].to_numpy())
    b_prime = Data(X = None, D=b.obsm['spatial'], Label=b.obs['leiden'].to_numpy())
    # calculate the loss between b_prime and b_prime
    loss_D, accuracy = loss(b_prime, b_prime)
    print(f"Loss: {loss_D}, Accuracy: {accuracy}")
    # Run FGW 2D alignment
    fgw_2d_work_dir = os.path.join(work_dir, 'fgw_2d')
    if not os.path.exists(fgw_2d_work_dir):
        os.makedirs(fgw_2d_work_dir)
    a_prime = fgw_2d_alignment(a, b_prime, fgw_2d_work_dir)
    fgw_2d_loss = loss(a_prime, b_prime)
    print(f"FGW 2D: loss: {fgw_2d_loss}")

    # Run ICP 2D alignment
    icp_2d_work_dir = os.path.join(work_dir, 'icp_2d')
    if not os.path.exists(icp_2d_work_dir):
        os.makedirs(icp_2d_work_dir)

    a_prime = icp_2d_alignment(a, b_prime, icp_2d_work_dir)
    icp_2d_loss = loss(a_prime, b_prime)
    print(f"ICP 2D: loss: {icp_2d_loss}")

    # Run direct alignment
    direct_alignment_work_dir = os.path.join(work_dir, 'direct_alignment')
    if not os.path.exists(direct_alignment_work_dir):
        os.makedirs(direct_alignment_work_dir)
    a_prime = direct_alignment(a, b_prime, direct_alignment_work_dir)
    direct_alignment_loss = loss(a_prime, b_prime)
    print(f"Direct alignment w/ center and scale: loss: {direct_alignment_loss}")

    # Run direct alignment without center and scale
    direct_alignment_work_dir = os.path.join(work_dir, 'direct_alignment_no_center_and_scale')
    if not os.path.exists(direct_alignment_work_dir):
        os.makedirs(direct_alignment_work_dir)

    a_prime = direct_alignment(a, b_prime, direct_alignment_work_dir, enable_center_and_scale=False)
    direct_alignment_loss = loss(a_prime, b_prime)
    print(f"Direct alignment w/o center and scale: loss: {direct_alignment_loss}")
    # Run HACCA
    # hacca(a, b, work_dir)