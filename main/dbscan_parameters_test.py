import open3d as o3d
import numpy as np
from sklearn import metrics
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import os

# results: 0.6 9 0.9; 1.1 26 0.8
def load_the_point_cloud(i, n, m):
    # true labels
    data_array = np.loadtxt("E:/3D-point-clouds-plant-phenotyping/data/A2_20220525_a.xyz", comments="//")
    labels_available = data_array.shape[1] == 8
    if labels_available:
        labels = data_array[:,6:]
    # 
    pcd = o3d.io.read_point_cloud("E:/3D-point-clouds-plant-phenotyping/output/original_data/point_cloud"+str(i)+".ply")
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    features = np.hstack((points*n, colors*m))
    return pcd, points, colors, features, labels

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
        if cluster_points:  # If the cluster is not empty
            cluster_points, cluster_colors = zip(*cluster_points)
            cluster_pcd = o3d.geometry.PointCloud()
            cluster_pcd.points = o3d.utility.Vector3dVector(np.array(cluster_points))
            cluster_pcd.colors = o3d.utility.Vector3dVector(np.array(cluster_colors))
            
            file_path = os.path.join(output_dir, f"point_cloud_cluster_{str(i)}-{label}.ply")
            o3d.io.write_point_cloud(file_path, cluster_pcd)
            print(f"Saved cluster {label} with {len(cluster_points)} points to {file_path}")

def visualise_results(pcd, labels):
    # Visualize the result (optional)
    max_label = labels.max()
    colormap = plt.get_cmap("tab20")
    colors = colormap(labels / (max_label if max_label > 0 else 1))
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd], window_name='DBSCAN clustering', width=800, height=600)

def purity_score(y_true, y_pred):
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 

if __name__ == "__main__":
    ### eps and min_samples
    # for i in np.arange(0.1, 11, 0.5):
    #     for j in range(2, 21, 2):
    ### colors and points distribution
    for i in np.arange(0.5, 1, 0.1):
        j = 1 - i
        pcd, points, colors, features, true_labels = load_the_point_cloud(i=16, n=i, m=j)
        labels, clusters = apply_DBSCAN(points=points, colors=colors, features=features, eps=0.6, min_samples=4)
        # save_sub_point_clouds(clusters=clusters, i=16)
        # visualise_results(pcd=pcd, labels=labels)
        score = purity_score(true_labels, labels)
        print(f"score {i}-{j}: {score}.")