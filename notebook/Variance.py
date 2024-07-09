from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import scanpy as sc
from sklearn.cross_decomposition import CCA
from sklearn.manifold import TSNE
from scipy.sparse import csr_matrix
from scipy.sparse import csr_matrix
import numpy as np
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

def var_select(df, m1, m2, n):
    # 1. 筛选出列均值在 m1 和 m2 之间的行
    filtered_df = df[(df.mean() > m1) & (df.mean() < m2)]

    # 2. 计算每行的方差，并选择前 n 个最大值所对应的索引
    variances = filtered_df.var(axis=0)/filtered_df.mean()
    sorted_indices = variances.sort_values(ascending=False).index
    selected_indices = sorted_indices[:n]

    return selected_indices.tolist()


def convert_to_array(x):
    if isinstance(x, csr_matrix):
        return x.toarray()
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise ValueError("Unsupported input type. Must be csr_matrix or ndarray.")

def tpm_normalize(counts_matrix):
    # 计算每个细胞的总表达量
    total_counts_per_cell = counts_matrix.sum(axis=1)

    # 计算每个基因的 TPM
    tpm_matrix = (counts_matrix.div(total_counts_per_cell, axis=0)) * 1e4
    return tpm_matrix

scaler = StandardScaler()
adata2 = sc.read_h5ad("I:\\mutiomics\\10X_Visium_Ratz2022Clonal_GSM4644086_10xvisium_data.h5ad")
X_data2 = convert_to_array(adata2.X)
X_data2 = pd.DataFrame(X_data2)
X_data2 = tpm_normalize(X_data2)
#X_data2 = np.log2(X_data2 + 1)
#X_data2 = scaler.fit_transform(X_data2)
X_data2 = pd.DataFrame(X_data2)
xx = pd.DataFrame([X_data2.mean(),X_data2.var()])

for m1 in np.arange(1, 2, 0.01):
    high_variance_variables = var_select(X_data2, m1, 20, 2000)
    gene2 = adata2.var.iloc[high_variance_variables, :]
    print(f"m1 = {m1}")
    print(gene2["highly_variable"].value_counts())

print(gene2["highly_variable"].value_counts())