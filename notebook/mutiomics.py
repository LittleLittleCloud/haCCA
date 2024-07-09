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

# 读取 H5AD 文件
adata1 = sc.read_h5ad("I:\\mutiomics\\MALDI_MALDI_brain_GruppeC_165x189_50umE0_data.h5ad")
adata2 = sc.read_h5ad("I:\\mutiomics\\10X_Visium_Ratz2022Clonal_GSM4644086_10xvisium_data.h5ad")
# 打印标题


data1_pca_results = adata1.obsm['X_umap']  # PCA 结果
pca_leiden = adata1.obs["leiden"]
data1_leiden_str = pca_leiden.to_numpy()

data2_pca_results = adata2.obsm['X_umap']  # PCA 结果
pca_leiden = adata2.obs["leiden"]
data2_leiden_str = pca_leiden.to_numpy()


# 打印转换后的 Series 对象
print(data1_leiden_str)
# 绘制空间图
selected_indices = np.random.choice(30000, 3734, replace=False)
plt.figure(figsize=(8, 6))
plt.scatter(data1_pca_results[:, 0], data1_pca_results[:, 1], c=data1_leiden_str.astype(int), cmap='tab20', s=20, alpha=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Spatial Distribution with PCA Coloring')
plt.colorbar(label='PCA 1')
plt.show()





def var_select(df, m1, m2, n):
    # 1. 筛选出列均值在 m1 和 m2 之间的行
    filtered_df = df.loc[:,(df.mean() > m1) & (df.mean() < m2)]

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

# 示例用法
# 假设 x 是 csr_matrix 或者 ndarray 对象
# 将 x 转换为 numpy.ndarray


#X_data = convert_to_array(adata1.X)
#high_variance_variables = var_select(X_data, 500)
#X_data = X_data[:,high_variance_variables]
#print(X_data.shape)
#pca = PCA()
# 对数据进行 PCA
#X_pca = pca.fit_transform(X_data)
#print(X_pca)

# 打印转换后的 Series 对象
#print(leiden_str)
# 绘制空间图
#plt.figure(figsize=(8, 6))
#plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data1_leiden_str.astype(int), cmap='tab20', s=20)
#plt.xlabel('X')
#plt.ylabel('Y')
#plt.title('Spatial Distribution with PCA Coloring')
#plt.colorbar(label='PCA 1')
#plt.show()

###umap
#umap_model = umap.UMAP(n_neighbors=10, min_dist=0.1, n_components=2)
#umap_results = umap_model.fit_transform(X_pca)

# 绘制降维结果的散点图
#plt.figure(figsize=(8, 6))
#plt.scatter(umap_results[:, 0], umap_results[:, 1], s=20)
#plt.xlabel('UMAP Component 1')
#plt.ylabel('UMAP Component 2')
#plt.title('UMAP Visualization')
#plt.show()



##tsne

#tsne_emb = TSNE(n_components=2,perplexity=10).fit_transform(X_pca[:,0:30])

# 绘制 t-SNE 结果
#plt.figure(figsize=(8, 6))
##plt.scatter(tsne_emb[:, 0], tsne_emb[:, 1], c=leiden_str.astype(int), cmap='tab20', s=20)
#plt.xlabel('t-SNE 1')
#plt.ylabel('t-SNE 2')
#plt.title('t-SNE Visualization')
#plt.show()

def tpm_normalize(counts_matrix):
    # 计算每个细胞的总表达量
    total_counts_per_cell = counts_matrix.sum(axis=1)
    # 计算每个基因的 TPM
    tpm_matrix = (counts_matrix.div(total_counts_per_cell, axis=0)) * 1e4
    return tpm_matrix

####adata1+adata2_cca(PCA first)
scaler = StandardScaler()
X_data1 = convert_to_array(adata1.X)
X_data1 = pd.DataFrame(X_data1)
X_data1 = X_data1.iloc[selected_indices, :]
X_data1 = scaler.fit_transform(X_data1)
X_data1 = pd.DataFrame(X_data1)
high_variance_variables = var_select(X_data1,-10,400, 500)
X_data1 = X_data1.iloc[:,high_variance_variables]
pca = PCA()
# 对数据进行 PCA
X1_pca = pca.fit_transform(X_data1)
umap_model = umap.UMAP(n_components=2, random_state=42)
X_1_umap_results = umap_model.fit_transform(X1_pca)
#X1_tsne = TSNE(n_components=3,perplexity=10, random_state=42).fit_transform(X1_pca[:,0:30])

X_data3 = convert_to_array(adata1.X)
X_data3 = pd.DataFrame(X_data3)
X_data3 = scaler.fit_transform(X_data3)
X_data3 = pd.DataFrame(X_data3)
X_data3 = X_data3.iloc[:,high_variance_variables]
pca = PCA()
# 对数据进行 PCA
X3_pca = pca.fit_transform(X_data3)
umap_model = umap.UMAP(n_components=2, random_state=42)
X_3_umap_results = umap_model.fit_transform(X3_pca)

X_data2 = convert_to_array(adata2.X)
X_data2 = pd.DataFrame(X_data2)
X_data2 = tpm_normalize(X_data2)
X_data2 = np.log2(X_data2 + 1)
X_data2 = pd.DataFrame(X_data2)
high_variance_variables = var_select(X_data2,0.01,40 ,2000)
X_data2 = X_data2.iloc[:,high_variance_variables]
#X_data2 = scaler.fit_transform(X_data2)
pca = PCA()
# 对数据进行 PCA
X2_pca = pca.fit_transform(X_data2)
umap_model = umap.UMAP(n_components=2, random_state=42)
X_2_umap_results = umap_model.fit_transform(X2_pca[:, 0:30])
#X2_tsne = TSNE(n_components=3,perplexity=10, random_state=42).fit_transform(X2_pca[:,0:30])



cca = CCA(n_components=2)
cca.fit(X_1_umap_results,X_2_umap_results)
X_c = cca.transform(X_1_umap_results)
Y_c = cca.transform(X_2_umap_results)
Z_c = cca.transform(X_3_umap_results)



print(X_c.shape)
import matplotlib.pyplot as plt


#X2_tsne = TSNE(n_components=3,perplexity=10, random_state=42).fit_transform(X2_pca[:,0:30])


# 假设 X_c 和 Y_c 是 CCA 转换后的结果
# 在这个示例中，假设 X_c 和 Y_c 都是二维数据

plt.figure(figsize=(15, 12))

# 绘制 X1 的散点图
plt.subplot(3, 2, 1)
plt.scatter(X_1_umap_results[:, 0], X_1_umap_results[:, 1], c=data1_leiden_str[selected_indices].astype(int), cmap='tab20', s=20, alpha=0.5)
plt.title('CCA Result for X1')
plt.xlabel('CCA Component 1')
plt.ylabel('CCA Component 2')

# 绘制 X1 转换后的散点图
plt.subplot(3, 2, 2)
plt.scatter(X_c[:, 0], X_c[:, 1], c=data1_leiden_str[selected_indices].astype(int), cmap='tab20', s=20, alpha=0.5)
plt.title('CCA Result for X1 (Transformed)')
plt.xlabel('CCA Component 1')
plt.ylabel('CCA Component 2')

# 绘制 X2 的散点图
plt.subplot(3, 2, 3)
plt.scatter(X_2_umap_results[:, 0], X_2_umap_results[:, 1], c=data2_leiden_str.astype(int), cmap='tab20', s=20, alpha=0.5)
plt.title('CCA Result for X2')
plt.xlabel('CCA Component 1')
plt.ylabel('CCA Component 2')

# 绘制 X2 转换后的散点图
plt.subplot(3, 2, 4)
plt.scatter(Y_c[:, 0], Y_c[:, 1], c=data2_leiden_str.astype(int), cmap='tab20', s=20, alpha=0.5, marker='s')
plt.title('CCA Result for X2 (Transformed)')
plt.xlabel('CCA Component 1')
plt.ylabel('CCA Component 2')

# 绘制 X3 的散点图
plt.subplot(3, 2, 5)
plt.scatter(X_3_umap_results[:, 0], X_3_umap_results[:, 1], c=data1_leiden_str.astype(int), cmap='tab20', s=20, alpha=0.5, marker='^')
plt.title('CCA Result for X3')
plt.xlabel('CCA Component 1')
plt.ylabel('CCA Component 2')

# 绘制 X3 转换后的散点图
plt.subplot(3, 2, 6)
plt.scatter(Z_c[:, 0], Z_c[:, 1], c=data1_leiden_str.astype(int), cmap='tab20', s=20, alpha=0.5, marker='^')
plt.title('CCA Result for X3 (Transformed)')
plt.xlabel('CCA Component 1')
plt.ylabel('CCA Component 2')

plt.tight_layout()
plt.show()


# 绘制 X_c 和 Y_c 的散点图
plt.figure(figsize=(8, 6))
plt.scatter(X_1_umap_results[:, 0], X_1_umap_results[:, 1], c=data1_leiden_str[selected_indices].astype(int), cmap='tab20', s=20, alpha=0.5)
plt.xlabel('CCA Component 1')
plt.ylabel('CCA Component 2')
plt.title('CCA Result for X')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(X_c[:, 0], X_c[:, 1], c=data1_leiden_str[selected_indices].astype(int), cmap='tab20', s=20, alpha=0.5)
plt.xlabel('CCA Component 1')
plt.ylabel('CCA Component 2')
plt.title('CCA Result for X')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(X_2_umap_results[:, 0], X_2_umap_results[:, 1], c=data2_leiden_str.astype(int), cmap='tab20', s=20, alpha=0.5)
plt.xlabel('CCA Component 1')
plt.ylabel('CCA Component 2')
plt.title('CCA Result for X')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(Y_c[:, 0], Y_c[:, 1], c=data2_leiden_str.astype(int), cmap='tab20', s=20, alpha=0.5)
plt.xlabel('CCA Component 1')
plt.ylabel('CCA Component 2')
plt.title('CCA Result for X')
plt.legend()
plt.show()

plt.figure(figsize=(10,10))
plt.scatter(X_3_umap_results[:, 0], X_3_umap_results[:, 1], c=data1_leiden_str.astype(int), cmap='tab20', s=20, alpha=0.5)
plt.scatter(X_2_umap_results[:, 0], X_2_umap_results[:, 1], c=data2_leiden_str.astype(int), cmap='tab20', s=20, alpha=0.5,marker='s')
plt.xlabel('CCA Component 1')
plt.ylabel('CCA Component 2')
plt.title('CCA Result for X')
plt.legend()
plt.show()

plt.figure(figsize=(10,10))
plt.scatter(Z_c[:, 0], Z_c[:, 1], c=data1_leiden_str.astype(int), cmap='tab20', s=20, alpha=0.5)
plt.scatter(Y_c[:, 0], Y_c[:, 1], c=data2_leiden_str.astype(int), cmap='tab20', s=20, alpha=0.5,marker='s')
plt.xlabel('CCA Component 1')
plt.ylabel('CCA Component 2')
plt.title('CCA Result for X')
plt.legend()
plt.show()



tsne_emb = TSNE(n_components=2,perplexity=10).fit_transform(X_c[:,0:30])
# 绘制 t-SNE 结果
plt.figure(figsize=(8, 6))
plt.scatter(tsne_emb[:, 0], tsne_emb[:, 1], c=leiden_str[selected_indices].astype(int), cmap='tab20', s=20)
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('t-SNE Visualization')
plt.show()

tsne_emb = TSNE(n_components=2,perplexity=10).fit_transform(X1_pca[:,0:30])
# 绘制 t-SNE 结果
plt.figure(figsize=(8, 6))
plt.scatter(tsne_emb[:, 0], tsne_emb[:, 1], c=leiden_str[selected_indices].astype(int), cmap='tab20', s=20, alpha=0)
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('t-SNE Visualization')
plt.show()


##############adata1+adata2_cca(no PCA first)
X_data1 = convert_to_array(adata1.X)
X_data1 = X_data1[selected_indices, :]
high_variance_variables = var_select(X_data1, 500)
X_data1 = X_data1[:,high_variance_variables]

X_data2 = convert_to_array(adata2.X)
high_variance_variables = var_select(X_data2, 500)
X_data2 = X_data2[:,high_variance_variables]

cca = CCA(n_components=10)
cca.fit(X_data1,X_data2)
X_c, Y_c = cca.transform(X_data1, X_data2)

import matplotlib.pyplot as plt

# 假设 X_c 和 Y_c 是 CCA 转换后的结果
# 在这个示例中，假设 X_c 和 Y_c 都是二维数据

# 绘制 X_c 和 Y_c 的散点图
plt.figure(figsize=(8, 4))
plt.scatter(X_c[:, 0], X_c[:, 1], c=leiden_str[selected_indices].astype(int), cmap='tab20', s=20, alpha=0.5)
plt.scatter(Y_c[:, 0], Y_c[:, 1], alpha=0.5)
plt.xlabel('CCA Component 1')
plt.ylabel('CCA Component 2')
plt.title('CCA Result for X')
plt.legend()

plt.show()

