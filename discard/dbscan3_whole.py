import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import os

# Load the point cloud
pcd = o3d.io.read_point_cloud("E:/3D-point-clouds-plant-phenotyping/output/original_data/point_cloud0.ply")

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

# Assign labels to the point cloud
max_label = labels.max()
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

# Visualize the result
o3d.visualization.draw_geometries([pcd], window_name='DBSCAN clustering', width=800, height=600)

# Optional: save the labeled point cloud
o3d.io.write_point_cloud("E:/3D-point-clouds-plant-phenotyping/output/labeled_data/point_cloud1.ply", pcd)
