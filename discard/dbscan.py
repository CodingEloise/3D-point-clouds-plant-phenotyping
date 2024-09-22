import os
import sys
import numpy as np
import open3d as o3d
import sklearn
import sklearn.cluster
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from assign_color import ColorAssigner

current_dir = os.path.dirname(os.path.abspath(__file__))

# load point cloud (already processed and saved)
pc_dir = os.path.join(current_dir, "..", "output", "original_data")
pc_file = "point_cloud18.ply"
pc_path = os.path.join(pc_dir, pc_file)
pc = o3d.io.read_point_cloud(pc_path)

# pre-process feature
def load_feature(pc, sample_size=10000):
	xyz = np.asarray(pc.points)
	rgb = np.asarray(pc.colors)

	if xyz.shape[0] > sample_size:
		indices = np.random.choice(xyz.shape[0], sample_size, replace=False)
		xyz = xyz[indices]
		rgb = rgb[indices]

	scaler = MinMaxScaler()
	xyz_scaled = scaler.fit_transform(xyz)
	rgb_scaled = scaler.fit_transform(rgb)
	weight_rgb = 0.85  # Higher weight for RGB
	weight_xyz = 0.15
	combined_data = np.hstack((weight_rgb * rgb_scaled, weight_xyz * xyz_scaled))
	return combined_data

# apply dbscan
train_features = load_feature(pc)
dbscan = sklearn.cluster.DBSCAN(eps=3, min_samples=2000)
labels = dbscan.fit_predict(train_features)

# evaluate results (over segmentation permitted)

# assign color and visualise (use method in a class)
color_assigner = ColorAssigner(pc, labels)
pc = color_assigner.assign_color_to_point_cloud()
o3d.visualization.draw_geometries([pc])