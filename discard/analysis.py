import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

class PointCloudAnalyzer:
    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.scans = self._get_scan_files()

    def _get_scan_files(self):
        return sorted([f for f in os.listdir(self.data_dir) if f.endswith(".xyz")])

    def load_point_cloud(self, file_path):
        data = np.loadtxt(file_path, comments='//')
        labels_available = data.shape[1] == 8
        return data, labels_available

    def segment_point_cloud(self, labels):
        x_labels = labels[:, 0]
        segments = {label: np.where(x_labels == label)[0] for label in np.unique(x_labels)}
        return segments

    def calculate_geometric_properties(self, points):
        hull = ConvexHull(points[:, :2])
        width = np.max(points[:, 0]) - np.min(points[:, 0])
        height = np.max(points[:, 1]) - np.min(points[:, 1])
        return hull.area, hull.volume, width / height if height != 0 else np.nan

    def plot_and_save(self, data, ylabel, title, filename):
        plt.figure(figsize=(10, 6))
        plt.plot(data)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

    def analyze_and_plot(self, start_index, end_index):
        for i in range(start_index, end_index):
            file_path = os.path.join(self.data_dir, self.scans[i])
            data, labels_available = self.load_point_cloud(file_path)
            if not labels_available:
                continue

            points = data[:, :3]
            colors = data[:, 3:6] / 255.0
            labels = data[:, 6:8]
            segments = self.segment_point_cloud(labels)

            for label, indices in segments.items():
                segment_points = points[indices]
                segment_colors = colors[indices]
                color_distribution = np.unique(segment_colors, axis=0, return_counts=True)[1]

                area, perimeter, aspect_ratio = self.calculate_geometric_properties(segment_points)

                self.plot_and_save(color_distribution, 'Count', f'Color Distribution for Label {label}', f'color_distribution_{label}.png')
                self.plot_and_save([area], 'Area', f'Geometric Properties for Label {label}', f'geometric_properties_{label}.png')

if __name__ == '__main__':
    data_dir = './data'
    output_dir = './output/B/'
    analyzer = PointCloudAnalyzer(data_dir, output_dir)
    analyzer.analyze_and_plot(14, 56)
