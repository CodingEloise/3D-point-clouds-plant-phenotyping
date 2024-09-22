import glob
import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

def save_dbscan_visualization(points, labels, eps, min_samples, point_cloud_name, output_dir):
    """
    保存DBSCAN聚类的可视化结果到指定的文件夹中.

    参数:
    - points: 点云坐标数组 (N, 3)
    - labels: DBSCAN 聚类后的标签数组 (N,)
    - eps: DBSCAN 的 eps 参数
    - min_samples: DBSCAN 的 min_samples 参数
    - point_cloud_name: 点云文件的名称，用于命名保存的图像文件
    - output_dir: 保存图像的文件夹路径
    """
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')
    
    max_label = labels.max()
    colormap = plt.get_cmap("tab20")
    colors = colormap(labels / (max_label if max_label > 0 else 1))
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1, cmap='tab20')

    ax.set_title(f'DBSCAN Clustering (eps={eps}, min_samples={min_samples})')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    os.makedirs(output_dir, exist_ok=True)

    file_name = f"{point_cloud_name}_eps_{eps}_min_samples_{min_samples}.png"
    file_path = os.path.join(output_dir, file_name)

    plt.savefig(file_path)
    plt.close()  # Close the figure to avoid memory issues

    print(f"Saved visualization to {file_path}")


# Load the point cloud
ply_files = glob.glob("E:/3D-point-clouds-plant-phenotyping/output/2nd_labeled_data/*.ply")
for ply_file in ply_files:
	point_cloud_name = os.path.splitext(os.path.basename(ply_file))[0]
	pcd = o3d.io.read_point_cloud(ply_file)

	points = np.asarray(pcd.points)
	colors = np.asarray(pcd.colors)
	colors = mcolors.rgb_to_hsv(colors)

	features = np.hstack((points, colors))

	for i in np.arange(0.6, 0.8, 0.1):
		for j in range(2, 7, 1):
			print(f"eps={i},min_samples={j}")
			# Apply DBSCAN
			eps = i  
			min_samples = j
			db = DBSCAN(eps=eps, min_samples=min_samples).fit(features)

			labels = db.labels_

			n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
			n_noise_ = list(labels).count(-1)

			print(f"Estimated number of clusters: {n_clusters_}")
			print(f"Estimated number of noise points: {n_noise_}")

			clusters = {label: [] for label in set(labels)}

			for point, color, label in zip(points, colors, labels):
				if label != -1:  # Ignore noise points
					clusters[label].append((point, color))

			output_dir = "E:/3D-point-clouds-plant-phenotyping/output/visualisation/"
			os.makedirs(output_dir, exist_ok=True)

			# Save each cluster as a separate point cloud file
			# for label, cluster_points in clusters.items():
			# 	if cluster_points:  # If the cluster is not empty
			# 		cluster_points, cluster_colors = zip(*cluster_points)
			# 		cluster_pcd = o3d.geometry.PointCloud()
			# 		cluster_pcd.points = o3d.utility.Vector3dVector(np.array(cluster_points))
			# 		cluster_pcd.colors = o3d.utility.Vector3dVector(np.array(cluster_colors))
					
			# 		file_path = os.path.join(output_dir, f"point_cloud_cluster_{str(i)}-{label}.ply")
			# 		o3d.io.write_point_cloud(file_path, cluster_pcd)
			# 		print(f"Saved cluster {label} with {len(cluster_points)} points to {file_path}")

			# Visualize the result (optional)
			# max_label = labels.max()
			# colormap = plt.get_cmap("tab20")
			# colors = colormap(labels / (max_label if max_label > 0 else 1))
			# pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
			# o3d.visualization.draw_geometries([pcd], window_name='DBSCAN clustering', width=800, height=600)

			save_dbscan_visualization(points, labels, eps, min_samples, point_cloud_name, output_dir)

