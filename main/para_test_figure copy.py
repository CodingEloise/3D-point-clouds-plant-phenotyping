import matplotlib.pyplot as plt
import numpy as np
import os
import open3d as o3d
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

# 获取点云文件
point_cloud_path = r'E:\3D-point-clouds-plant-phenotyping\output\original_data\point_cloud12.ply'
pcd = o3d.io.read_point_cloud(point_cloud_path)

# 创建输出文件夹
output_dir = 'dbscan_visualizations'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 定义 DBSCAN 参数
params = [[0.1, 20], [0.6, 9], [5, 20]]

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

# 获取点云文件
point_cloud_path = r'E:\3D-point-clouds-plant-phenotyping\output\original_data\point_cloud12.ply'
pcd = o3d.io.read_point_cloud(point_cloud_path)

# 转换点云数据为numpy数组
xyz = np.asarray(pcd.points)

# 创建输出文件夹
output_dir = 'dbscan_visualizations'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 定义参数组合
params = [
    (0.6, 4),
    (5, 20),
    (0.6, 8)
]

for eps, min_samples in params:
    # 归一化数据
    scaler = MinMaxScaler()
    xyz_scaled = scaler.fit_transform(xyz)
    
    # 执行DBSCAN聚类
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(xyz_scaled)

    # 统计簇数量和噪点数量
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    # 可视化
    unique_labels = set(labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = 'k'  # 噪点用黑色表示
        class_mask = (labels == k)
        plt.scatter(xyz[class_mask, 0], xyz[class_mask, 1], c=col, s=0.1)

    plt.title(f'DBSCAN (eps={eps}, min_samples={min_samples})\nClusters: {n_clusters}, Noise: {n_noise}')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    
    # 保存可视化图像
    plt.savefig(os.path.join(output_dir, f'dbscan_eps{eps}_min{min_samples}.png'))
    plt.close()