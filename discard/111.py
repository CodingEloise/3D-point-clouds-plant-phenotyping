import os
import colour
import numpy as np
import open3d as o3d
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt

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

def filter_points_by_label(pc, labels, target_label):
    # Filters out points that match the target label.
    indices = np.where(labels == target_label)[0]
    filtered_pc = pc.select_by_index(indices)
    return filtered_pc

def load_feature(pc):
    xyz = np.asarray(pc.points)
    rgb = np.asarray(pc.colors)
    scaler = MinMaxScaler()
    xyz_scaled = scaler.fit_transform(xyz)
    rgb_scaled = scaler.fit_transform(rgb)
    weight_rgb = 0.70  # Adjusted weight for RGB
    weight_xyz = 0.30  # Adjusted weight for XYZ
    combined_data = np.hstack((weight_rgb * rgb_scaled, weight_xyz * xyz_scaled))
    return combined_data

data_dir = '/mnt/data/'  # Adjusted path for uploaded file
all_files = sorted(os.listdir(data_dir))
scans = [fname for fname in all_files if fname.endswith(".xyz")]

results = []

# For loop over selected point cloud files
for i in [0, 14, 80]:
    pcd_path = os.path.join(data_dir, scans[i])
    pointcloud, labels_available, true_labels = load_as_o3d_cloud(pcd_path)
    
    # Step 1: Use KMeans to cluster based on Y axis to separate table, packaging, and plants
    y_coords = np.asarray(pointcloud.points)[:, 1].reshape(-1, 1)
    kmeans_y = KMeans(n_clusters=3, random_state=24, n_init=20, max_iter=300)
    labels_y = kmeans_y.fit_predict(y_coords)
    
    # Assuming the cluster with the lowest mean Y coordinate is the table, the highest is the plants, and the middle one is the packaging
    cluster_means = [np.mean(y_coords[labels_y == k]) for k in range(3)]
    table_cluster = np.argmin(cluster_means)
    plant_cluster = np.argmax(cluster_means)
    
    # Filter out points for plants
    plant_pc = filter_points_by_label(pointcloud, labels_y, plant_cluster)
    
    # Step 2: Use KMeans to cluster the plants based on combined features
    combined_data = load_feature(plant_pc)
    
    for k in range(2, 10):  # Start from 2 because silhouette score is not defined for a single cluster
        kmeans = KMeans(n_clusters=k, random_state=24, n_init=20, max_iter=300)
        labels = kmeans.fit_predict(combined_data)
        
        # Ensure the number of colors matches the number of clusters
        unique_labels = np.unique(labels)
        num_clusters = len(unique_labels)
        colors = generate_distinct_color(num_clusters)
        
        # Assign colors to each cluster
        colored_labels = np.array([colors[label] if label >= 0 else [0, 0, 0] for label in labels])
        plant_pc.colors = o3d.utility.Vector3dVector(colored_labels)
        o3d.visualization.draw_geometries([plant_pc])

        # Subset data for silhouette score calculation
        subset_size = 10000  # Adjust subset size based on your data size and performance
        if combined_data.shape[0] > subset_size:
            subset_indices = np.random.choice(combined_data.s
