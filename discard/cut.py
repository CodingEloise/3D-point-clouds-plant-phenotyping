import os
import numpy as np
import open3d as o3d
import colour
import time

class PointCloudProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.scans = self._get_scan_files()
        self.output_dir = './output/B/'
        self._create_output_dir()

    def _get_scan_files(self):
        """Get all .xyz files in the data directory."""
        return sorted([f for f in os.listdir(self.data_dir) if f.endswith(".xyz")])

    def _create_output_dir(self):
        """Create the main output directory if it does not exist."""
        os.makedirs(self.output_dir, exist_ok=True)

    def load_as_array(self, file_path):
        """Load point cloud data from an .xyz file into a numpy array."""
        data_array = np.loadtxt(file_path, comments='//')
        labels_available = data_array.shape[1] == 8
        return data_array, labels_available

    def segment_point_cloud(self, labels):
        """Segment the point cloud into parts based on the x component of labels."""
        x_labels = labels[:, 0]
        unique_labels = np.unique(x_labels)
        segments = {label: [] for label in unique_labels}

        for i, label in enumerate(x_labels):
            segments[label].append(i)

        return segments

    def load_as_o3d_cloud(self, file_path):
        """Load point cloud data into an Open3D point cloud object and segment it."""
        data, labels_available = self.load_as_array(file_path)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(data[:, :3])

        if labels_available:
            labels = data[:, 6:8]
            segments = self.segment_point_cloud(labels)
            return pc, segments, labels_available
        else:
            pc.colors = o3d.utility.Vector3dVector(data[:, 3:6] / 255)
            return pc, None, labels_available

    def save_segmented_clouds(self, pointcloud, segments, base_dir):
        """Save each segment of the point cloud separately in corresponding subdirectories."""
        for label, indices in segments.items():
            segment_pc = pointcloud.select_by_index(indices)
            label_dir = os.path.join(base_dir, str(int(label)))
            os.makedirs(label_dir, exist_ok=True)
            output_file = os.path.join(label_dir, f"segment_{int(label)}_{int(time.time())}.ply")
            o3d.io.write_point_cloud(output_file, segment_pc)

    def process_scans(self, start_index, end_index):
        """Process and save segmented point clouds within the specified index range."""
        for i in range(start_index, end_index):
            file_path = os.path.join(self.data_dir, self.scans[i])

            # Process parts only (based on x component of labels)
            pointcloud_parts, segments_parts, labels_available_parts = self.load_as_o3d_cloud(file_path)
            if labels_available_parts:
                self.save_segmented_clouds(pointcloud_parts, segments_parts, os.path.join(self.output_dir, "parts_only"))

if __name__ == '__main__':
    data_dir = './data'
    processor = PointCloudProcessor(data_dir)
    processor.process_scans(14, 56)
