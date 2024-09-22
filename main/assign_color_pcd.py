import colour
import numpy as np
import open3d as o3d

class ColorAssigner:
	def __init__(self, point_cloud, labels) -> None:
		self.pc = point_cloud
		self.labels = np.array(labels)
		self.num = self._get_number_of_unique_labels()
		
	def _get_number_of_unique_labels(self):
		unique_labels = np.unique(self.labels)
		num = len(unique_labels)
		return num

	def assign_color_to_point_cloud(self):
		RGB = np.zeros((self.num, 3))
		for i in range(self.num):
			hue = i / self.num
			hsl = (hue, 1, 0.5) # high contract colors
			rgb = colour.models.HSL_to_RGB(hsl)
			RGB[i] = rgb

		print(f"Unique labels: {self.num}")
		print(f"RGB array: {RGB}")
	
		labeled_colors = np.array([RGB[label] if label >= 0 else [0, 0, 0] for label in self.labels])
		self.pc.colors = o3d.utility.Vector3dVector(labeled_colors)
		print(f"Labeled colors: {labeled_colors[:10]}")

		return self.pc

	def visualise_point_cloud(self):
		pass

	def save_point_cloud(self):
		pass

if __name__ == '__main__':
	points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1]
    ])
	pc = o3d.geometry.PointCloud()
	pc.points = o3d.utility.Vector3dVector(points)
	labels = [0,1,2,3,4]
	color_assigner = ColorAssigner(pc, labels)
	pc = color_assigner.assign_color_to_point_cloud()
	o3d.visualization.draw_geometries([pc])
