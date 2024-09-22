# this file is to test over different k values and color- distribution on the results
import os
import colour
import numpy as np
import open3d as o3d
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt

# Set the environment variable to suppress the warning
os.environ["COLOUR_SCIENCE__COLOUR__IMPORT_VAAB_COLOUR"] = "True"

def generate_distinct_color(total_colors):
	"""Generates a distinct color based on the index and total number of colors."""
	RGB = np.zeros((total_colors, 3))
	for i in range(total_colors):
		hue = i / total_colors
		hsl = (hue, 1, 0.5)  # High saturation and medium lightness for vibrant colors
		rgb = colour.models.HSL_to_RGB(hsl)
		RGB[i] = rgb
	return RGB

def load_as_array(path):
	# Loads the data from an .xyz file into a numpy array.
	# Also returns a boolean indicating whether per-point labels are available.
	data_array = np.loadtxt(path, comments='//')  # raw data as np array, of shape (nx6) or (nx8) if labels are available.
	labels_available = data_array.shape[1] == 8
	return data_array, labels_available

def load_as_o3d_cloud(path):
	# Loads the data from an .xyz file into an open3d point cloud object.
	data, labels_available = load_as_array(path)
	pc = o3d.geometry.PointCloud()
	pc.points = o3d.utility.Vector3dVector(data[:, 0:3])
	pc.colors = o3d.utility.Vector3dVector(data[:, 3:6] / 255)
	labels = None
	if labels_available:
		labels = data[:, 6].astype(int)  # Ensure labels are integers
	return pc, labels_available, labels

def load_feature(pc):
	xyz = np.asarray(pc.points)
	rgb = np.asarray(pc.colors)
	scaler = MinMaxScaler()
	xyz_scaled = scaler.fit_transform(xyz)
	rgb_scaled = scaler.fit_transform(rgb)
	weight_rgb = 0.85  # Higher weight for RGB
	weight_xyz = 0.15
	combined_data = np.hstack((weight_rgb * rgb_scaled, weight_xyz * xyz_scaled))
	return combined_data

data_dir = 'E:/3D-point-clouds-plant-phenotyping/data/'
output_dir = 'E:/3D-point-clouds-plant-phenotyping/output/'
os.makedirs(output_dir, exist_ok=True)
all_files = sorted(os.listdir(data_dir))
scans = [fname for fname in all_files if fname.endswith(".xyz")]

# Store all results for plotting
all_results = []

# For loop over selected point cloud files
for i in [0, 14, 24, 34, 44, 54, 64, 74, 80]:
	pcd_path = os.path.join(data_dir, scans[i])
	pointcloud, labels_available, true_labels = load_as_o3d_cloud(pcd_path)
	
	combined_data = load_feature(pointcloud)
	results = []
	
	for k in range(2, 10):  # Start from 2 because silhouette score is not defined for a single cluster
		kmeans = KMeans(n_clusters=k, random_state=24, n_init=20, max_iter=300)
		labels = kmeans.fit_predict(combined_data)
		
		# Ensure the number of colors matches the number of clusters
		unique_labels = np.unique(labels)
		num_clusters = len(unique_labels)
		colors = generate_distinct_color(num_clusters)
		
		# Assign colors to each cluster
		colored_labels = np.array([colors[label] if label >= 0 else [0, 0, 0] for label in labels])
		pointcloud.colors = o3d.utility.Vector3dVector(colored_labels)
		# o3d.visualization.draw_geometries([pointcloud])
		
		subset_size = 50000  # Adjust subset size based on your data size and performance
		if combined_data.shape[0] > subset_size:
			subset_indices = np.random.choice(combined_data.shape[0], subset_size, replace=False)
			subset_data = combined_data[subset_indices]
			subset_labels = labels[subset_indices]
		else:
			subset_data = combined_data
			subset_labels = labels

		# Calculate silhouette score for the entire dataset
		silhouette_avg = silhouette_score(subset_data, subset_labels)
		print(f'{i}-{k}, Silhouette Score: {silhouette_avg}')
		results.append((k, silhouette_avg))
		
		filename = os.path.join(output_dir, f"point_cloud_k_means_{i}-{k}.ply")
		# o3d.io.write_point_cloud(filename, pointcloud)
		# print(f"Point cloud saved to {filename}")
	
	all_results.append((i, results))

# Plot the results
plt.figure(figsize=(10, 8))
for i, results in all_results:
	n_clusters, silhouette_scores = zip(*results)
	plt.plot(n_clusters, silhouette_scores, marker='o', label=f'Point Cloud {i}')
plt.title('Silhouette Score vs. Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'silhouette_scores_comparison.png'))
plt.show()
