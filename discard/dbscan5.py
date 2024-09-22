import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree
from joblib import Parallel, delayed
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# 加载点云数据
pcd = o3d.io.read_point_cloud("E:/3D-point-clouds-plant-phenotyping/output/original_data/point_cloud" + str(16) + ".ply")

# 提取点坐标和颜色信息
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)
features = np.hstack((points, colors))

# 设置抽样比例
sample_ratio = 0.1
sample_size = int(features.shape[0] * sample_ratio)

# 随机抽样
indices = np.random.choice(features.shape[0], sample_size, replace=False)
sampled_features = features[indices]

# 自定义距离函数
def custom_distance(point1, point2):
    coord_distance = np.linalg.norm(point1[:3] - point2[:3])
    color_distance = np.linalg.norm(point1[3:6] - point2[3:6])
    return coord_distance * 0.5 + color_distance * 0.5

# 构建 BallTree
print("构建 BallTree...")
tree = BallTree(sampled_features, metric=custom_distance)

# 并行计算邻域信息
def compute_neighborhood(i):
    return tree.query_radius([sampled_features[i]], r=3.0, return_distance=True)

print("计算邻域...")
results = Parallel(n_jobs=-1, verbose=10)(delayed(compute_neighborhood)(i) for i in range(sampled_features.shape[0]))

# 提取距离和索引
distances = [result[0][0] for result in results]
indices = [result[1][0] for result in results]

# 初始化 DBSCAN
print("运行 DBSCAN 聚类...")
eps = 3.0  # 根据数据调整
min_samples = 20  # 根据数据调整
db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')

# 构建距离矩阵
n_samples = sampled_features.shape[0]
distance_matrix = np.zeros((n_samples, n_samples))

for i in tqdm(range(n_samples), desc="构建距离矩阵"):
    distance_matrix[i, indices[i]] = distances[i]

# 运行 DBSCAN 聚类
db.fit(distance_matrix)

# 提取聚类标签
labels = db.labels_

# 统计聚类结果
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print(f"Estimated number of clusters: {n_clusters_}")
print(f"Estimated number of noise points: {n_noise_}")

# 保存和可视化结果
clusters = {label: [] for label in set(labels)}
for point, color, label in zip(sampled_features[:, :3], sampled_features[:, 3:], labels):
    if label != -1:  # 忽略噪声点
        clusters[label].append((point, color))

output_dir = "E:/3D-point-clouds-plant-phenotyping/output/labeled_data/"
os.makedirs(output_dir, exist_ok=True)

for label, cluster_points in clusters.items():
    if cluster_points:  # 如果聚类不为空
        cluster_points, cluster_colors = zip(*cluster_points)
        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(np.array(cluster_points))
        cluster_pcd.colors = o3d.utility.Vector3dVector(np.array(cluster_colors))
        
        file_path = os.path.join(output_dir, f"point_cloud_cluster_{str(16)}-{label}.ply")
        o3d.io.write_point_cloud(file_path, cluster_pcd)
        print(f"Saved cluster {label} with {len(cluster_points)} points to {file_path}")

# 可视化结果
max_label = labels.max()
colormap = plt.get_cmap("tab20")
colors = colormap(labels / (max_label if max_label > 0 else 1))
sampled_pcd = o3d.geometry.PointCloud()
sampled_pcd.points = o3d.utility.Vector3dVector(sampled_features[:, :3])
sampled_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([sampled_pcd], window_name='DBSCAN clustering', width=800, height=600)
