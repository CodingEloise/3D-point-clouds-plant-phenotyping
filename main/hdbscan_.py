import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans

# 创建最终保存点云的文件夹
final_folder = 'final'
if not os.path.exists(final_folder):
    os.makedirs(final_folder)

def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return np.asarray(pcd.points), np.asarray(pcd.colors), pcd

def apply_dbscan(features, eps, min_samples):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(features)
    return labels

def apply_cut_at_max_density(points, axis=2, bins=50, shift=0.01):
    hist, bin_edges = np.histogram(points[:, axis], bins=bins)
    max_density_index = np.argmax(hist)
    cut_value = (bin_edges[max_density_index] + bin_edges[max_density_index + 1]) / 2
    cut_value += shift
    labels = np.zeros(points.shape[0])
    labels[points[:, axis] >= cut_value] = 1
    return labels, cut_value

def cluster_and_color_point_cloud(file_path, color_weight, coord_weight, dbscan_eps, dbscan_min_samples, dbscan_eps_inner=0.8, dbscan_min_samples_inner=2, index=0):
    points, colors, pcd = load_point_cloud(file_path)
    final_label = np.zeros(points.shape[0])

    initial_labels = apply_dbscan(points, eps=dbscan_eps, min_samples=dbscan_min_samples)
    final_label[initial_labels != -1] = initial_labels[initial_labels != -1] + 1

    unique_labels, counts = np.unique(initial_labels, return_counts=True)
    max_label = unique_labels[np.argmax(counts)]

    cut_points = points[initial_labels == max_label]
    cut_labels, cut_value = apply_cut_at_max_density(cut_points, axis=2, shift=0.01)

    cut_labels_full = np.full(initial_labels.shape, -1)
    cut_labels_full[initial_labels == max_label] = cut_labels

    final_label[cut_labels_full != -1] = cut_labels_full[cut_labels_full != -1] + np.max(final_label)

    for i in np.unique(cut_labels):
        dbscan_points = points[cut_labels_full == i]
        dbscan_features = np.hstack((dbscan_points * coord_weight, colors[cut_labels_full == i] * color_weight))
        dbscan_labels = apply_dbscan(dbscan_features, eps=dbscan_eps_inner, min_samples=dbscan_min_samples_inner)
        final_label[cut_labels_full == i] = dbscan_labels + np.max(final_label)

    max_combined_label = int(np.max(final_label))
    color_map = np.random.rand(max_combined_label + 1, 3)
    colored_points = color_map[final_label.astype(int)]
    pcd.colors = o3d.utility.Vector3dVector(colored_points)

    final_file_path = os.path.join(final_folder, f"colored_point_cloud_{index}.ply")
    o3d.io.write_point_cloud(final_file_path, pcd)
    print(f"Saved final colored point cloud to {final_file_path}")

    # 保存二维图像
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colored_points, s=1)
    ax.view_init(elev=45, azim=-45)  # 设置视角为45度
    image_file_path = os.path.join(final_folder, f"point_cloud_image_{index}.png")
    plt.savefig(image_file_path)
    plt.close()
    print(f"Saved 2D image of point cloud to {image_file_path}")

# 批量处理点云
for i in range(0, 82):
    cluster_and_color_point_cloud(
        file_path=f"E:/3D-point-clouds-plant-phenotyping/output/original_data/point_cloud{i}.ply",
        color_weight=0.4,
        coord_weight=0.6,
        dbscan_eps=1,
        dbscan_min_samples=5,
        dbscan_eps_inner=0.8,
        dbscan_min_samples_inner=2,
        index=i
    )
