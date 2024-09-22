import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import os

# Load the point cloud
for i in range(81):
	pcd = o3d.io.read_point_cloud("E:/3D-point-clouds-plant-phenotyping/output/original_data/point_cloud"+str(i)+".ply")

	# Extract point coordinates and colors
	points = np.asarray(pcd.points)
	colors = np.asarray(pcd.colors)

	# Combine spatial and color information
	# Normalizing the color values (assuming colors are in [0, 1])
	features = np.hstack((points, colors))

	# Apply DBSCAN
	eps = 3  # Choose based on your data scale
	min_samples = 20  # Choose based on your data density
	db = DBSCAN(eps=eps, min_samples=min_samples).fit(features)

	# Extract labels
	labels = db.labels_

	# Number of clusters in labels, ignoring noise if present
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
	n_noise_ = list(labels).count(-1)

	print(f"Estimated number of clusters: {n_clusters_}")
	print(f"Estimated number of noise points: {n_noise_}")

	# Create a dictionary to hold the points of each cluster
	clusters = {label: [] for label in set(labels)}

	# Separate points into different clusters
	for point, color, label in zip(points, colors, labels):
		if label != -1:  # Ignore noise points
			clusters[label].append((point, color))

	# Create directory to save the sub-point clouds if not exists
	output_dir = "E:/3D-point-clouds-plant-phenotyping/output/labeled_data/"
	os.makedirs(output_dir, exist_ok=True)

	# Save each cluster as a separate point cloud file
	for label, cluster_points in clusters.items():
		if cluster_points:  # If the cluster is not empty
			cluster_points, cluster_colors = zip(*cluster_points)
			cluster_pcd = o3d.geometry.PointCloud()
			cluster_pcd.points = o3d.utility.Vector3dVector(np.array(cluster_points))
			cluster_pcd.colors = o3d.utility.Vector3dVector(np.array(cluster_colors))
			
			file_path = os.path.join(output_dir, f"point_cloud_cluster_{str(i)}-{label}.ply")
			o3d.io.write_point_cloud(file_path, cluster_pcd)
			print(f"Saved cluster {label} with {len(cluster_points)} points to {file_path}")

	# Visualize the result (optional)
	max_label = labels.max()
	colormap = plt.get_cmap("tab20")
	colors = colormap(labels / (max_label if max_label > 0 else 1))
	pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
	# o3d.visualization.draw_geometries([pcd], window_name='DBSCAN clustering', width=800, height=600)
