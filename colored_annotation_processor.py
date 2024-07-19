import os
import numpy as np
import open3d as o3d
import colour
import time

class PointCloudProcessor:
	def __init__(self, data_dir):
		self.data_dir = data_dir
		self.scans = self._get_scan_files()
		self.output_dirs = {
			"parts_only": './output/parts_only/',
			"parts_and_instances": './output/parts_and_instances/'
		}
		self._create_output_dirs()

	def _get_scan_files(self):
		all_files = sorted(os.listdir(self.data_dir))
		return [fname for fname in all_files if fname.endswith(".xyz")]

	def _create_output_dirs(self):
		for dir_name in self.output_dirs.values():
			os.makedirs(dir_name, exist_ok=True)

	def load_as_array(self, file_path):
		"""Loads the data from an .xyz file into a numpy array."""
		data_array = np.loadtxt(file_path, comments='//')
		labels_available = data_array.shape[1] == 8
		return data_array, labels_available

	def generate_distinct_color(self, index, total_colors):
		"""Generates a distinct color based on the index and total number of colors."""
		hue = index / total_colors
		hsl = (hue, 1, 0.5)  # High saturation and medium lightness for vibrant colors
		rgb = colour.models.HSL_to_RGB(hsl)
		return rgb

	def assign_distinct_color(self, labels, class_only):
		"""Generate unique colors for each distinct part or part+instance combination."""
		colors = np.zeros((labels.shape[0], 3))
		label_colors = {}

		combined_labels = [str(label[0]) if class_only else f"{label[0]}-{label[1]}" for label in labels]
		unique_labels = np.unique(combined_labels)
		total_labels = len(unique_labels)

		for index, label in enumerate(unique_labels):
			label_colors[label] = self.generate_distinct_color(index, total_labels)

		for i, label in enumerate(combined_labels):
			colors[i] = label_colors[label]

		return colors

	def load_as_o3d_cloud(self, file_path, class_only):
		"""Loads the data from an .xyz file into an open3d point cloud object."""
		data, labels_available = self.load_as_array(file_path)
		pc = o3d.geometry.PointCloud()
		pc.points = o3d.utility.Vector3dVector(data[:, 0:3])

		if labels_available:
			labels = data[:, 6:]
			colors = self.assign_distinct_color(labels, class_only)
			pc.colors = o3d.utility.Vector3dVector(colors)
		else:
			pc.colors = o3d.utility.Vector3dVector(data[:, 3:6] / 255)

		return pc, labels_available

	def visualize_and_save_point_cloud(self, pc, filename, auto_close=True):
		"""Optionally use o3d.visualization.Visualize() to automatically close the windows."""
		o3d.io.write_point_cloud(filename, pc)
		print(f"Point cloud saved to {filename}")
		if auto_close:
			vis = o3d.visualization.Visualizer()
			vis.create_window()
			vis.add_geometry(pc)
			vis.poll_events()
			vis.update_renderer()
			time.sleep(2)
			vis.destroy_window()
		else:
			o3d.visualization.draw_geometries([pc])

	def process_scans(self, start_index, end_index):
		for i in range(start_index, end_index):
			file_path = os.path.join(self.data_dir, self.scans[i])

			# Process parts only
			pointcloud_parts, labels_available_parts = self.load_as_o3d_cloud(file_path, class_only=True)
			if labels_available_parts:
				output_file_parts = os.path.join(self.output_dirs["parts_only"], f"point_cloud_parts_{i}.ply")
				self.visualize_and_save_point_cloud(pointcloud_parts, output_file_parts, auto_close=True)

			# Process parts + instances
			pointcloud_instances, labels_available_instances = self.load_as_o3d_cloud(file_path, class_only=False)
			if labels_available_instances:
				output_file_instances = os.path.join(self.output_dirs["parts_and_instances"], f"point_cloud_instances_{i}.ply")
				self.visualize_and_save_point_cloud(pointcloud_instances, output_file_instances, auto_close=True)

if __name__ == '__main__':
	data_dir = './data'
	processor = PointCloudProcessor(data_dir)
	processor.process_scans(14, 56)
