import open3d as o3d
import numpy as np
from sklearn import metrics
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import os

def load_the_point_cloud(i):
	pcd = o3d.io.read_point_cloud("E:/3D-point-clouds-plant-phenotyping/output/original_data/point_cloud"+str(i)+".ply")
	points = np.asarray(pcd.points)
	colors = np.asarray(pcd.colors)
	features = np.hstack((points*0.7, colors*0.3))
	return pcd, points, colors, features

def load_labeled_data(data_dir, data):
	# true labels
	data_array = np.loadtxt(data_dir + data, comments='//')
	labels_available = data_array.shape[1] == 8
	if labels_available:
		labels = data_array[:,6:]
	# point cloud
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(data_array[:,0:3])
	pcd.colors = o3d.utility.Vector3dVector(data_array[:,3:6]/255)
	points = np.asarray(pcd.points)
	colors = np.asarray(pcd.colors)
	#
	features = np.hstack((points*0.7, colors*0.3))
	return pcd, points, colors, features, labels

def get_all_data():
	data_dir = 'E:/3D-point-clouds-plant-phenotyping/labeled_data/'
	all_files = sorted(os.listdir(data_dir))
	scans = [fname for fname in all_files if fname.endswith(".xyz")]
	return data_dir, scans

def apply_DBSCAN(points, colors, features, eps, min_samples):
	eps = eps  # Choose based on your data scale
	min_samples = min_samples  # Choose based on your data density
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
    # Visualize the result (optional)
    max_label = labels.max()
    colormap = plt.get_cmap("tab20")
    colors = colormap(labels / (max_label if max_label > 0 else 1))
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='DBSCAN clustering', width=800, height=600)
    vis.add_geometry(pcd)

    # 设置固定视角
    view_ctl = vis.get_view_control()
    # 通过设置目标点、距离、俯仰角和方位角来调整视角
    view_ctl.set_front([0.577, -0.577, 0.3])  # 45度的方向向量
    view_ctl.set_lookat([0, 0, 0])  # 视点中心
    view_ctl.set_up([0, 0, 1])  # 向上的方向
    view_ctl.set_zoom(0.5)  # 适当的缩放

    vis.run()
    vis.destroy_window()


def purity_score(y_true, y_pred):
	if y_true.ndim > 1:
		y_true = np.argmax(y_true, axis=1)
	# compute contingency matrix (also called confusion matrix)
	contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
	# return purity
	return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 

def plot_noise_trend(params, noise_data):
		n_values = [p[0] for p in params]
		eps_values = [p[1] for p in params]
		min_samples_values = [p[2] for p in params]

		plt.figure(figsize=(10, 6))
		plt.scatter(range(len(noise_data)), noise_data, c=eps_values, cmap='viridis', label='Noise Points')
		plt.colorbar(label='Epsilon (eps)')
		plt.xlabel('Parameter Combination Index')
		plt.ylabel('Noise Points Count')
		plt.title('Noise Points Count Trend for Different Parameter Combinations')
		plt.show()

def plot_purity_score(params, purity_scores):
	eps_values = [p[1] for p in params]
	min_samples_values = [p[2] for p in params]

	plt.figure(figsize=(10, 6))
	plt.scatter(range(len(purity_scores)), purity_scores, c=eps_values, cmap='plasma', label='Purity Score')
	plt.colorbar(label='Epsilon (eps)')
	plt.xlabel('Parameter Combination Index')
	plt.ylabel('Purity Score')
	plt.title('Purity Score Trend for Different Parameter Combinations')
	plt.show()

if __name__ == "__main__":
	noise_data = []
	purity_scores = []
	params = []

	data_dir, scans = get_all_data()
	for n in range(13):
		for i in [0.1, 1, 5]:
			for j in [2, 9, 20]:
				pcd, points, colors, features, true_labels = load_labeled_data(data_dir=data_dir, data=scans[n])
				labels, clusters = apply_DBSCAN(points=points, colors=colors, features=features, eps=i, min_samples=j)
				visualise_results(pcd=pcd, labels=labels)
				
				score = purity_score(true_labels, labels)
				n_noise_ = list(labels).count(-1)
				print(f"pcd-{n} eps-{i} min_samples-{j} purity score: {score}.")
				noise_data.append(n_noise_)
				purity_scores.append(score)
				params.append((n, i, j))

	plot_noise_trend(params, noise_data)
	plot_purity_score(params, purity_scores)