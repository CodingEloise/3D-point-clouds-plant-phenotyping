from matplotlib import pyplot as plt
import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
import os

def slice_point_cloud(points, axis, intervals):
    slices = []
    for min_val, max_val in intervals:
        if axis == 'z':
            mask = (points[:, 2] >= min_val) & (points[:, 2] < max_val)
        elif axis == 'y':
            mask = (points[:, 1] >= min_val) & (points[:, 1] < max_val)
        elif axis == 'x':
            mask = (points[:, 0] >= min_val) & (points[:, 0] < max_val)
        slices.append(points[mask])
    return slices

def dbscan_clustering(slice_points, eps=3, min_samples=20):
    if slice_points.shape[0] == 0:  # If no points in this slice, skip
        return np.array([])
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(slice_points)
    return db.labels_

def combine_clusters(z_clusters, x_clusters, y_clusters):
    # Simple example of majority voting or any other merging strategy
    combined_labels = np.full_like(z_clusters[0], -1)
    num_points = len(z_clusters[0])
    
    for i in range(num_points):
        label_votes = [z_clusters[0][i], x_clusters[0][i], y_clusters[0][i]]
        valid_votes = [label for label in label_votes if label != -1]
        if valid_votes:
            combined_labels[i] = max(set(valid_votes), key=valid_votes.count)
    
    return combined_labels

# Load the point cloud and process in slices
for i in range(81):
    pcd = o3d.io.read_point_cloud(f"E:/3D-point-clouds-plant-phenotyping/output/original_data/point_cloud{i}.ply")

    # Extract point coordinates and colors
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # Define intervals for slicing along different axes
    num_slices = 10  # Number of slices
    z_intervals = np.linspace(points[:, 2].min(), points[:, 2].max(), num_slices + 1)
    x_intervals = np.linspace(points[:, 0].min(), points[:, 0].max(), num_slices + 1)
    y_intervals = np.linspace(points[:, 1].min(), points[:, 1].max(), num_slices + 1)

    # Perform clustering along Z, X, and Y axes separately
    z_slices = slice_point_cloud(points, 'z', zip(z_intervals[:-1], z_intervals[1:]))
    x_slices = slice_point_cloud(points, 'x', zip(x_intervals[:-1], x_intervals[1:]))
    y_slices = slice_point_cloud(points, 'y', zip(y_intervals[:-1], y_intervals[1:]))

    z_clusters = [dbscan_clustering(slice_points) for slice_points in z_slices]
    x_clusters = [dbscan_clustering(slice_points) for slice_points in x_slices]
    y_clusters = [dbscan_clustering(slice_points) for slice_points in y_slices]

    # Combine the clusters from different axes
    final_clusters = combine_clusters(z_clusters, x_clusters, y_clusters)

    # Number of clusters in final_clusters, ignoring noise if present
    n_clusters_ = len(set(final_clusters)) - (1 if -1 in final_clusters else 0)
    n_noise_ = list(final_clusters).count(-1)
    
    print(f"Estimated number of clusters: {n_clusters_}")
    print(f"Estimated number of noise points: {n_noise_}")

    # Create a dictionary to hold the points of each cluster
    clusters = {label: [] for label in set(final_clusters)}

    # Separate points into different clusters
    for point, color, label in zip(points, colors, final_clusters):
        if label != -1:  # Ignore noise points
            clusters[label].append((point, color))

    # Create directory to save the sub-point clouds if not exists
    output_dir = "E:/3D-point-clouds-plant-phenotyping/output/labeled_data1/"
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

    # Optional: Visualize the merged result
    max_label = final_clusters.max()
    colormap = plt.get_cmap("tab20")
    final_colors = colormap(final_clusters / (max_label if max_label > 0 else 1))
    pcd.colors = o3d.utility.Vector3dVector(final_colors[:, :3])
    # o3d.visualization.draw_geometries([pcd], window_name='DBSCAN clustering', width=800, height=600)
