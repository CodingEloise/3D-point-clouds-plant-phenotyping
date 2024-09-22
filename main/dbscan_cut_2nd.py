import glob
import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

# 25 out of 82
best_params = {
    "point_cloud_cluster_0-0": {"eps": 0.7, "min_samples": 5},
    "point_cloud_cluster_0-1": {"eps": 0.8, "min_samples": 2},
    "point_cloud_cluster_1-0": {"eps": 0.8, "min_samples": 2},
    "point_cloud_cluster_2-0": {"eps": 0.8, "min_samples": 2},
    "point_cloud_cluster_3-1": {"eps": 0.8, "min_samples": 2},
    "point_cloud_cluster_4-1": {"eps": 0.8, "min_samples": 2},
    "point_cloud_cluster_14-1": {"eps": 0.8, "min_samples": 2},
    "point_cloud_cluster_15-0": {"eps": 0.8, "min_samples": 2},
    "point_cloud_cluster_1-1": {"eps": 0.8, "min_samples": 5},
    "point_cloud_cluster_2-1": {"eps": 0.8, "min_samples": 6, "note": "not good"},
    "point_cloud_cluster_3-0": {"eps": 0.7, "min_samples": 5},
    "point_cloud_cluster_4-0": {"eps": 0.7, "min_samples": 5},
    "point_cloud_cluster_10-0": {"eps": 0.8, "min_samples": 5},
    "point_cloud_cluster_10-1": {"eps": 0.8, "min_samples": 2},
    "point_cloud_cluster_11-0": {"eps": 0.8, "min_samples": 2, "note": "not good"},
    "point_cloud_cluster_11-1": {"eps": 0.8, "min_samples": 3},
    "point_cloud_cluster_12-0": {"eps": 0.8, "min_samples": 3},
    "point_cloud_cluster_12-1": {"eps": 0.8, "min_samples": 6},
    "point_cloud_cluster_13-0": {"eps": 0.6, "min_samples": 2},
    "point_cloud_cluster_13-1": {"eps": 1, "min_samples": 5, "note": "doesn't work even with eyes"},
    "point_cloud_cluster_14-0": {"eps": 0.8, "min_samples": 6},
    "point_cloud_cluster_15-1": {"eps": 0.7, "min_samples": 2},
    "point_cloud_cluster_16-0": {"eps": 0.8, "min_samples": 2, "note": "not good"},
    "point_cloud_cluster_16-1": {"eps": 0.7, "min_samples": 5},
    "point_cloud_cluster_17-0": {"eps": 0.7, "min_samples": 5},
    "point_cloud_cluster_17-1": {"eps": 0.8, "min_samples": 3},
    "point_cloud_cluster_18-0": {"eps": 0.8, "min_samples": 2, "note": "not good"},
    "point_cloud_cluster_18-1": {"eps": 0.8, "min_samples": 4},
    "point_cloud_cluster_19-0": {"eps": 0.8, "min_samples": 2},
    "point_cloud_cluster_19-1": {"eps": 0.8, "min_samples": 2},
    "point_cloud_cluster_20-0": {"eps": 0.8, "min_samples": 2, "note": "try other params"},
    "point_cloud_cluster_20-1": {"eps": 0.8, "min_samples": 2, "note": "not good"},
    "point_cloud_cluster_21-0": {"eps": 0.8, "min_samples": 2},
    "point_cloud_cluster_21-1": {"eps": 0.8, "min_samples": 6, "note": "try 7 or more"},
    "point_cloud_cluster_22-0": {"eps": 0.8, "min_samples": 5},
    "point_cloud_cluster_22-1": {"eps": 0.7, "min_samples": 4},
    "point_cloud_cluster_23-0": {"eps": 0.6, "min_samples": 3, "note": "not good"},
    "point_cloud_cluster_23-1": {"eps": 0.6, "min_samples": 2},
    "point_cloud_cluster_24-0": {"eps": 0.8, "min_samples": 5},
    "point_cloud_cluster_24-1": {"eps": 0.8, "min_samples": 2},
    "point_cloud_cluster_25-0": {"eps": 0.8, "min_samples": 2},
    "point_cloud_cluster_25-0": {"eps": 0.6, "min_samples": 3},
    "point_cloud_cluster_25-1": {"eps": 0.7, "min_samples": 5},
}

# Load the point cloud
ply_files = glob.glob("E:/3D-point-clouds-plant-phenotyping/output/2nd_labeled_data/*.ply")
for ply_file in ply_files:
     point_cloud_name = os.path.splitext(os.path.basename(ply_file))[0]
     pcd = o3d.io.read_point_cloud(ply_file)

     points = np.asarray(pcd.points)
     colors = np.asarray(pcd.colors)
     colors = mcolors.rgb_to_hsv(colors)

     features = np.hstack((points, colors))

     # Retrieve the best parameters for the current point cloud
     params = best_params.get(point_cloud_name, None)
     if params is None:
          print(f"No parameters found for {point_cloud_name}, skipping.")
          continue

     eps = params["eps"]
     min_samples = params["min_samples"]

     print(f"Processing {point_cloud_name} with eps={eps}, min_samples={min_samples}")

     # Apply DBSCAN
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

     output_dir = f"E:/3D-point-clouds-plant-phenotyping/output/3rd_labeled_data/{point_cloud_name}"
     os.makedirs(output_dir, exist_ok=True)
     
     min_points_threshold = 100

     # Save each cluster as a separate point cloud file
     for label, cluster_points in clusters.items():
          if len(cluster_points) > min_points_threshold:  # If the cluster is not empty
                cluster_points, cluster_colors = zip(*cluster_points)
                cluster_pcd = o3d.geometry.PointCloud()
                cluster_pcd.points = o3d.utility.Vector3dVector(np.array(cluster_points))
                cluster_pcd.colors = o3d.utility.Vector3dVector(np.array(cluster_colors))
                
                file_path = os.path.join(output_dir, f"{point_cloud_name}-{label}.ply")
                o3d.io.write_point_cloud(file_path, cluster_pcd)
                print(f"Saved cluster {label} with {len(cluster_points)} points to {file_path}")
          else:
                print(f"Cluster {label} has only {len(cluster_points)} points, not saving.")