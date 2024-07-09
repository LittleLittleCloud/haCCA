
from sklearn.cross_decomposition import CCA
import numpy as np

# 假设 X 和 Y 是两组数据，每一行代表一个样本
X = np.random.rand(100, 5)  # 100个样本，每个样本5个特征
Y = np.random.rand(100, 3)  # 100个样本，每个样本3个特征

# 创建 CCA 模型
cca = CCA(n_components=2)

# 训练模型
cca.fit(X, Y)

# 进行 CCA 转换
X_c, Y_c = cca.transform(X, Y)

import matplotlib.pyplot as plt

# 假设 X_c 和 Y_c 是 CCA 转换后的结果
# 在这个示例中，假设 X_c 和 Y_c 都是二维数据

# 绘制 X_c 和 Y_c 的散点图
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.scatter(X_c[:, 0], X_c[:, 1], c='r', label='X')
plt.xlabel('CCA Component 1')
plt.ylabel('CCA Component 2')
plt.title('CCA Result for X')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(Y_c[:, 0], Y_c[:, 1], c='b', label='Y')
plt.xlabel('CCA Component 1')
plt.ylabel('CCA Component 2')
plt.title('CCA Result for Y')
plt.legend()

plt.tight_layout()
plt.show()