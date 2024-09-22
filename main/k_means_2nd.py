import os
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

i = 0
for file in ['point_cloud_cluster_0-0.ply', 'point_cloud_cluster_1-2.ply', 'point_cloud_cluster_2-2.ply', 'point_cloud_cluster_3-2.ply', 'point_cloud_cluster_4-1.ply', 'point_cloud_cluster_5-2.ply', 'point_cloud_cluster_6-1.ply', 'point_cloud_cluster_7-3.ply', 'point_cloud_cluster_8-7.ply', 'point_cloud_cluster_9-2.ply', 'point_cloud_cluster_10-3.ply', 'point_cloud_cluster_11-0.ply', 'point_cloud_cluster_12-0.ply', 'point_cloud_cluster_13-0.ply', 'point_cloud_cluster_14-1.ply', 'point_cloud_cluster_15-0.ply', 'point_cloud_cluster_16-0.ply', 'point_cloud_cluster_17-4.ply', 'point_cloud_cluster_18-11.ply', 'point_cloud_cluster_19-5.ply', 'point_cloud_cluster_20-13.ply', 'point_cloud_cluster_21-2.ply', 'point_cloud_cluster_22-7.ply', 'point_cloud_cluster_23-4.ply', 'point_cloud_cluster_24-0.ply', 'point_cloud_cluster_25-0.ply', 'point_cloud_cluster_26-0.ply', 'point_cloud_cluster_27-0.ply', 'point_cloud_cluster_28-2.ply', 'point_cloud_cluster_29-1.ply', 'point_cloud_cluster_30-1.ply', 'point_cloud_cluster_31-0.ply', 'point_cloud_cluster_32-0.ply', 'point_cloud_cluster_33-2.ply', 'point_cloud_cluster_34-1.ply', 'point_cloud_cluster_35-4.ply', 'point_cloud_cluster_36-1.ply', 'point_cloud_cluster_37-4.ply', 'point_cloud_cluster_38-0.ply', 'point_cloud_cluster_39-0.ply', 'point_cloud_cluster_40-1.ply', 'point_cloud_cluster_41-0.ply', 'point_cloud_cluster_42-2.ply', 'point_cloud_cluster_43-0.ply', 'point_cloud_cluster_44-1.ply', 'point_cloud_cluster_45-1.ply', 'point_cloud_cluster_46-2.ply', 'point_cloud_cluster_47-1.ply', 'point_cloud_cluster_48-1.ply', 'point_cloud_cluster_49-1.ply', 'point_cloud_cluster_50-0.ply', 'point_cloud_cluster_51-0.ply', 'point_cloud_cluster_52-0.ply', 'point_cloud_cluster_53-0.ply', 'point_cloud_cluster_54-0.ply', 'point_cloud_cluster_55-0.ply', 'point_cloud_cluster_56-1.ply', 'point_cloud_cluster_57-1.ply', 'point_cloud_cluster_58-1.ply', 'point_cloud_cluster_59-0.ply', 'point_cloud_cluster_60-5.ply', 'point_cloud_cluster_61-2.ply', 'point_cloud_cluster_62-1.ply', 'point_cloud_cluster_63-3.ply', 'point_cloud_cluster_64-2.ply', 'point_cloud_cluster_65-1.ply', 'point_cloud_cluster_66-2.ply', 'point_cloud_cluster_67-0.ply', 'point_cloud_cluster_68-1.ply', 'point_cloud_cluster_69-0.ply', 'point_cloud_cluster_70-0.ply', 'point_cloud_cluster_71-3.ply', 'point_cloud_cluster_72-0.ply', 'point_cloud_cluster_73-1.ply', 'point_cloud_cluster_74-0.ply', 'point_cloud_cluster_75-1.ply', 'point_cloud_cluster_76-0.ply', 'point_cloud_cluster_77-1.ply', 'point_cloud_cluster_78-1.ply', 'point_cloud_cluster_79-3.ply', 'point_cloud_cluster_80-1.ply']:
	path = "E:/3D-point-clouds-plant-phenotyping/output/labeled_data"
	pcd = o3d.io.read_point_cloud(os.path.join(path, file))

	xyz = np.asarray(pcd.points)
	rgb = np.asarray(pcd.colors)
	hsv = mcolors.rgb_to_hsv(rgb)
	scaler = MinMaxScaler()
	xyz_scaled = scaler.fit_transform(xyz)
	rgb_scaled = scaler.fit_transform(hsv)
	# test over weights
	# for t in np.arange(0, 1, 0.1):
		# k = 1 - t
	weight_rgb = 0.4  # Higher weight for RGB
	weight_xyz = 0.6
	features = np.hstack((weight_rgb * rgb_scaled, weight_xyz * xyz_scaled))

	n_clusters = 2 
	kmeans = KMeans(n_clusters=n_clusters, n_init=20, max_iter=200)
	kmeans.fit(features)

	labels = kmeans.labels_

	clusters = {label: [] for label in set(labels)}
	for point, color, label in zip(xyz, rgb, labels):
		clusters[label].append((point, color))

	# Create output directory
	output_dir = "E:/3D-point-clouds-plant-phenotyping/output/2nd_labeled_data/"
	os.makedirs(output_dir, exist_ok=True)

	for label, cluster_points in clusters.items():
		if cluster_points:  # If the cluster is not empty
			cluster_points, cluster_colors = zip(*cluster_points)
			cluster_pcd = o3d.geometry.PointCloud()
			cluster_pcd.points = o3d.utility.Vector3dVector(np.array(cluster_points))
			cluster_pcd.colors = o3d.utility.Vector3dVector(np.array(cluster_colors))
			
			file_path = os.path.join(output_dir, f"point_cloud_cluster_{i}-{label}.ply")
			o3d.io.write_point_cloud(file_path, cluster_pcd)
			print(f"Saved cluster {label} with {len(cluster_points)} points to {file_path}")

	max_label = labels.max()
	colormap = plt.get_cmap("tab20")
	color_map = colormap(labels / (max_label if max_label > 0 else 1))
	pcd.colors = o3d.utility.Vector3dVector(color_map[:, :3])
	# o3d.visualization.draw_geometries([pcd], window_name='k-means clustering', width=800, height=600)

	i+=1
