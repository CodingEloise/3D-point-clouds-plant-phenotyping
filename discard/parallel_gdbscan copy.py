import open3d as o3d
import numpy as np
import multiprocessing
import os
import math
from gdbscan import GDBSCAN, Points


class Point:
    def __init__(self, x, y, z, r, g, b):
        self.x = x
        self.y = y
        self.z = z
        self.r = r
        self.g = g
        self.b = b
        self.cluster_id = -2

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


def load_the_point_cloud(i):
    data_array = np.loadtxt("E:/3D-point-clouds-plant-phenotyping/data/A2_20220525_a.xyz", comments='//')
    labels_available = data_array.shape[1] == 8
    if labels_available:
        labels = data_array[:, 6:]
    
    pcd = o3d.io.read_point_cloud("E:/3D-point-clouds-plant-phenotyping/output/original_data/point_cloud" + str(i) + ".ply")
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    features = np.hstack((points * 0.7, colors * 0.3))
    
    return pcd, points, colors, features, labels, data_array


def split_point_cloud(points, colors, features, data_array, n_splits):
    point_chunks = np.array_split(points, n_splits)
    color_chunks = np.array_split(colors, n_splits)
    feature_chunks = np.array_split(features, n_splits)
    data_chunks = np.array_split(data_array, n_splits)
    
    return point_chunks, color_chunks, feature_chunks, data_chunks


def run_gdbscan(points, colors, features, data_array, split_index):
    # 转换为 Point 对象
    point_list = [Point(*data[:6]) for data in data_array]
    clusters = GDBSCAN(Points(point_list), n_pred, 5, w_card)

    labels = [-1] * len(point_list)
    for cluster_id, cluster in enumerate(clusters):
        for point in cluster:
            index = point_list.index(point)
            labels[index] = cluster_id

    # 保存结果到文件
    np.savez(f"gdbscan_result_{split_index}.npz", labels=labels, points=points, colors=colors)
    
    return labels


def parallel_gdbscan(points, colors, features, data_array, n_splits):
    point_chunks, color_chunks, feature_chunks, data_chunks = split_point_cloud(points, colors, features, data_array, n_splits)
    
    processes = []
    
    for i in range(n_splits):
        p = multiprocessing.Process(target=run_gdbscan, args=(point_chunks[i], color_chunks[i], feature_chunks[i], data_chunks[i], i))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()


def merge_gdbscan_results(n_splits):
    all_labels = []
    all_points = []
    all_colors = []
    
    for i in range(n_splits):
        result = np.load(f"gdbscan_result_{i}.npz")
        all_labels.append(result['labels'])
        all_points.append(result['points'])
        all_colors.append(result['colors'])
    
    all_labels = np.concatenate(all_labels)
    all_points = np.concatenate(all_points)
    all_colors = np.concatenate(all_colors)
    
    return all_labels, all_points, all_colors


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
    
    o3d.visualization.draw_geometries([pcd], window_name='GDBSCAN clustering', width=800, height=600)


if __name__ == "__main__":
    pcd, points, colors, features, true_labels, data_array = load_the_point_cloud(i=16)
    
    # 选择拆分数
    n_splits = 4
    
    # 并行运行 GDBSCAN
    parallel_gdbscan(points, colors, features, data_array, n_splits=n_splits)
    
    # 合并结果
    labels, points, colors = merge_gdbscan_results(n_splits=n_splits)
    
    # 将结果应用到点云
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 可视化结果
    visualise_results(pcd, labels)
