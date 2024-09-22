import open3d as o3d
import numpy as np
from sklearn import metrics
from sklearn.cluster import HDBSCAN
import matplotlib.pyplot as plt
import os
# from gdbscan import GDBSCAN, Points
import math

UNCLASSIFIED = -2

class Point:
    def __init__(self, x, y, z, r, g, b):
        self.x = x
        self.y = y
        self.z = z
        self.r = r
        self.g = g
        self.b = b
        self.cluster_id = UNCLASSIFIED

    def __repr__(self):
        return '(x:{}, y:{}, z:{}, r:{}, g:{}, b:{}, cluster:{})'.format(
            self.x, self.y, self.z, self.r, self.g, self.b, self.cluster_id)

    def __eq__(self, other):
        if isinstance(other, Point):
            return (self.x, self.y, self.z, self.r, self.g, self.b) == (other.x, other.y, other.z, other.r, other.g, other.b)
        return False

    def __hash__(self):
        return hash((self.x, self.y, self.z, self.r, self.g, self.b))


def n_pred(p1, p2):
    spatial_distance = math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
    color_distance = math.sqrt((p1.r - p2.r)**2 + (p1.g - p2.g)**2 + (p1.b - p2.b)**2)
    return all([spatial_distance <= 2, color_distance <= 1])


def w_card(points):
    return len(list(points))

def load_the_point_cloud(i, sample_size=None):
    # true labels
    data_array = np.loadtxt("E:/3D-point-clouds-plant-phenotyping/data/A2_20220525_a.xyz", comments='//')
    
    # 采样指定数量的点
    if sample_size:
        data_array = data_array[:sample_size]
    
    labels_available = data_array.shape[1] == 8
    if labels_available:
        labels = data_array[:, 6:]
    
    pcd = o3d.io.read_point_cloud("E:/3D-point-clouds-plant-phenotyping/output/original_data/point_cloud" + str(i) + ".ply")
    
    # 仅保留采样的点
    points = np.asarray(pcd.points)[:len(data_array)]
    colors = np.asarray(pcd.colors)[:len(data_array)]
    features = np.hstack((points * 0.7, colors * 0.3))
    
    return pcd, points, colors, features, labels, data_array


def apply_GDBSCAN(sample_size=None):
    # 采样并加载点云数据
    pcd, points, colors, features, labels, data_array = load_the_point_cloud(16, sample_size=sample_size)
    
    # 创建 Point 对象的列表
    point_list = []
    for data in data_array:
        point_list.append(Point(*data[:6]))
    
    print("GDBSCAN starts.")
    clusters = GDBSCAN(Points(point_list), n_pred, 5, w_card)
    print("GDBSCAN ends.")
    
    # 初始化 labels 为未分类
    labels = [-1] * len(point_list)  # 假设未分类的标签是 -1

    # 遍历 clusters，并为每个点分配标签
    for cluster_id, cluster in enumerate(clusters):
        for point in cluster:
            index = point_list.index(point)
            labels[index] = cluster_id

    return labels, clusters, pcd

def apply_HDBSCAN(points, colors, features, min_cluster_size=100, min_samples=10):
    # 创建 HDBSCAN 聚类器
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    
    # 拟合特征并生成标签
    labels = clusterer.fit_predict(features)
    
    # 统计簇的数量和噪声点数量
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    print(f"Estimated number of clusters: {n_clusters_}")
    print(f"Estimated number of noise points: {n_noise_}")
    
    # 将每个点和对应的颜色按照标签分配到簇
    clusters = {label: [] for label in set(labels)}
    for point, color, label in zip(points, colors, labels):
        if label != -1:  # Ignore noise points
            clusters[label].append((point, color))
    
    return labels, clusters

def save_sub_point_clouds(clusters, i):
    # Create directory to save the sub-point clouds if not exists
    output_dir = "E:/3D-point-clouds-plant-phenotyping/output/labeled_data/"
    os.makedirs(output_dir, exist_ok=True)

    # Save each cluster as a separate point cloud file
    for label, cluster_points in clusters.items():
        if len(cluster_points) >= 10:  # If the cluster is not empty
            cluster_points, cluster_colors = zip(*cluster_points)
            cluster_pcd = o3d.geometry.PointCloud()
            cluster_pcd.points = o3d.utility.Vector3dVector(np.array(cluster_points))
            cluster_pcd.colors = o3d.utility.Vector3dVector(np.array(cluster_colors))
            
            file_path = os.path.join(output_dir, f"point_cloud_cluster_{i}-{label}.ply")
            o3d.io.write_point_cloud(file_path, cluster_pcd)
            print(f"Saved cluster {i}-{label} with {len(cluster_points)} points to {file_path}")
        else:
            print(f"Cluster {i}-{label} has only {len(cluster_points)} points and will not be saved.")


def visualise_results(pcd, labels):
    labels = np.array(labels)
    max_label = labels.max()
    
    if max_label == -1:
        max_label = 0
    
    normalized_labels = (labels - labels.min()) / (max_label - labels.min() + 1e-8)
    
    colormap = plt.get_cmap("tab20")
    colors = colormap(normalized_labels)
    
    colors = colors[:, :3]
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.visualization.draw_geometries([pcd], window_name='DBSCAN clustering', width=800, height=600)


def purity_score(y_true, y_pred):
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 


if __name__ == "__main__":
    sample_size = 500  # 修改这个值来采样更多或更少的点
    
    labels, clusters, pcd = apply_HDBSCAN(sample_size=sample_size)
    
   #  save_sub_point_clouds(clusters=clusters, i=16)
    
    visualise_results(pcd=pcd, labels=labels)
    
    pcd, points, colors, features, true_labels, _ = load_the_point_cloud(i=16, sample_size=sample_size)
    score = purity_score(true_labels, labels)
    print(f"score: {score}.")
