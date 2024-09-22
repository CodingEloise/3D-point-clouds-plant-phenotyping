import matplotlib.pyplot as plt
import numpy as np
import os
import open3d as o3d
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.colors as mcolors

# 获取点云文件
point_cloud_path = r'E:\3D-point-clouds-plant-phenotyping\output\original_data\point_cloud0.ply'
pcd = o3d.io.read_point_cloud(point_cloud_path)

# 创建一个文件夹来保存生成的图像
output_dir = 'visualizations'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 定义颜色:坐标比值与K值的组合
k_values = [2, 5, 8, 10]
color_coord_ratios = ["0:1", "0.2:0.8", "0.5:0.5", "0.8:0.2", "1:0"]

# 生成可视化图像
def generate_point_cloud_visualization(k, color_coord_ratio):
    # 获取点云数据
    xyz = np.asarray(pcd.points)
    rgb = np.asarray(pcd.colors)
    hsv = mcolors.rgb_to_hsv(rgb)

    # 特征缩放
    scaler = MinMaxScaler()
    xyz_scaled = scaler.fit_transform(xyz)
    rgb_scaled = scaler.fit_transform(hsv)

    # 权重设置
    weight_rgb, weight_xyz = map(float, color_coord_ratio.split(':'))
    total_weight = weight_rgb + weight_xyz
    weight_rgb /= total_weight  # 标准化权重
    weight_xyz /= total_weight
    
    features = np.hstack((weight_rgb * rgb_scaled, weight_xyz * xyz_scaled))

    # K-Means 聚类
    kmeans = KMeans(n_clusters=k, n_init=20, max_iter=200)
    kmeans.fit(features)

    labels = kmeans.labels_

    # 创建聚类字典
    clusters = {label: [] for label in set(labels)}
    for point, color, label in zip(xyz, rgb, labels):
        clusters[label].append((point, color))

    # 保存聚类结果
    for label, cluster_points in clusters.items():
        if cluster_points:  # 如果聚类不为空
            cluster_points, cluster_colors = zip(*cluster_points)
            cluster_pcd = o3d.geometry.PointCloud()
            cluster_pcd.points = o3d.utility.Vector3dVector(np.array(cluster_points))
            cluster_pcd.colors = o3d.utility.Vector3dVector(np.array(cluster_colors))
            
            file_path = os.path.join(output_dir, f"point_cloud_cluster_k{str(k)}_label{label}.ply")
            o3d.io.write_point_cloud(file_path, cluster_pcd)
            print(f"Saved cluster {label} with {len(cluster_points)} points to {file_path}")

    # 更新点云颜色以便可视化
    max_label = labels.max()
    colormap = plt.get_cmap("tab20")
    color_map = colormap(labels / (max_label if max_label > 0 else 1))
    pcd.colors = o3d.utility.Vector3dVector(color_map[:, :3])

    # 可视化
    o3d.visualization.draw_geometries([pcd], window_name=f'K-means clustering (k={k})', width=800, height=600)

# 逐个调用生成可视化图像
for k in k_values:
    for color_coord_ratio in color_coord_ratios:
        generate_point_cloud_visualization(k, color_coord_ratio)
