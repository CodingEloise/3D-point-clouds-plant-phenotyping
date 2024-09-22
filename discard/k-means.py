import os
import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import colour
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.spatial.distance import cdist

# Though the following import is not directly being used, it is required
# for 3D projection to work with matplotlib < 3.2
import mpl_toolkits.mplot3d  # noqa: F401

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
        labels = data[:, 6]  # Select the first column of labels if available
    return pc, labels_available, labels

def visualize_clusters(pointcloud, labels, n_clusters, title):
    """Visualizes the clusters with distinct colors."""
    colors = generate_distinct_color(n_clusters)
    cluster_colors = colors[labels]
    pointcloud.colors = o3d.utility.Vector3dVector(cluster_colors)
    o3d.visualization.draw_geometries([pointcloud], window_name=title)

def custom_distance_metric(x, y, weight_rgb, weight_xyz):
    """Custom distance metric that considers both color and spatial information."""
    return weight_rgb * np.linalg.norm(x[:3] - y[:3]) + weight_xyz * np.linalg.norm(x[3:] - y[3:])

data_dir = 'E:/intern/LAST-Straw/data/'

all_files = sorted(os.listdir(data_dir))
scans = [fname for fname in all_files if fname.endswith(".xyz")]

for i in range(84):
	# Ensure the point cloud and labels are loaded from the same file
	pcd_path = data_dir + scans[i]
	pointcloud, labels_available, true_labels = load_as_o3d_cloud(pcd_path)
	if true_labels is not None:
		true_labels = true_labels.astype(int)  # Ensure true_labels is integer type

# Use the same point cloud data for clustering and evaluation
xyz = np.asarray(pointcloud.points)
rgb = np.asarray(pointcloud.colors)

# Normalize the data using MinMaxScaler instead of StandardScaler
scaler = MinMaxScaler()
xyz_scaled = scaler.fit_transform(xyz)
rgb_scaled = scaler.fit_transform(rgb)

# Experiment with different weight combinations for RGB and XYZ
weight_rgb = 0.7  # Higher weight for RGB
weight_xyz = 0.3  # Lower weight for XYZ

combined_data = np.hstack((weight_rgb * rgb_scaled, weight_xyz * xyz_scaled))

random_seed = 42
results = []

# Determine the optimal number of clusters using the elbow method
wcss = []
range_k = range(1, 21)  # Adjust the range of k values

for k in range_k:
    kmeans = KMeans(n_clusters=k, random_state=21, n_init=20, max_iter=300)
    kmeans.fit(combined_data)
    
    wcss.append(kmeans.inertia_)
    print(kmeans.inertia_)
# Plot the elbow method results
plt.figure(figsize=(10, 5))
plt.plot(range_k, wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Within-cluster sum of squares (WCSS)')
plt.show()

# Calculate the difference in WCSS for successive values of k
diff_wcss = np.diff(wcss)
diff2_wcss = np.diff(diff_wcss)

# Find the index of the maximum second derivative (point of maximum curvature)
elbow_point = np.argmax(diff2_wcss) + 2  # Adding 2 to account for the diff offset

print(f'Optimal number of clusters: {elbow_point}')

# Focus on fewer clusters to capture semantic parts
for n in range(2, 22):  # Adjust the range to avoid too many clusters
    print(f'Starting KMeans for n_clusters={n}')
    try:
        kmeans = KMeans(n_clusters=n, random_state=random_seed, n_init=20, max_iter=300)
        labels = kmeans.fit_predict(combined_data)
        print(f'KMeans completed for n_clusters={n}')
        
        # Generate distinct colors for the clusters
        cluster_colors = generate_distinct_color(n)[labels]
        
        # Update the colors of the point cloud
        pointcloud.colors = o3d.utility.Vector3dVector(cluster_colors)
        
      #   Visualize the point cloud with the cluster colors
        o3d.visualization.draw_geometries([pointcloud], window_name=f"KMeans Clusters: {n}")
        print(f'Visualization completed for n_clusters={n}')
        
        # Calculate and print the silhouette score
        if labels_available:
            print(f'Evaluating Adjusted Rand Index for n_clusters={n}')
            if len(true_labels) == len(labels):
                ari_score = adjusted_rand_score(true_labels, labels)
                print(f'Number of clusters: {n}, Adjusted Rand Index: {ari_score}')
            else:
                print(f'Number of clusters: {n}, true_labels and labels length mismatch.')
        
        # Subset data for silhouette score calculation
        subset_size = 10000  # Adjust subset size based on your data size and performance
        if combined_data.shape[0] > subset_size:
            subset_indices = np.random.choice(combined_data.shape[0], subset_size, replace=False)
            subset_data = combined_data[subset_indices]
            subset_labels = labels[subset_indices]
        else:
            subset_data = combined_data
            subset_labels = labels
        
        # Use subset data for silhouette score calculation
        silhouette_avg = silhouette_score(subset_data, subset_labels)
        print(f'Number of clusters: {n}, Silhouette Score: {silhouette_avg}')
        
        # Store results for later analysis
        results.append((n, ari_score, silhouette_avg))
    except Exception as e:
        print(f'Error processing n_clusters={n}: {e}')
    print(f'Finished processing n_clusters={n}\n')

# Plot the results
n_clusters, ari_scores, silhouette_scores = zip(*results)
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.plot(n_clusters, ari_scores, marker='o')
plt.title('Adjusted Rand Index vs. Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Adjusted Rand Index')

plt.subplot(1, 2, 2)
plt.plot(n_clusters, silhouette_scores, marker='o')
plt.title('Silhouette Score vs. Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')

plt.tight_layout()
plt.show()

