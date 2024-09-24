import os
from matplotlib import colors, pyplot as plt
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

# 假设每个 .xyz 文件的格式是 x, y, z, r, g, b, label
i = 0
path = "E:/3D-point-clouds-plant-phenotyping/labeled_data"

for file in os.listdir(path):
    if file.endswith(".xyz"):
        # 读取点云数据并提取标签
        data = np.genfromtxt(os.path.join(path, file), comments='//', dtype=float)
        xyz = data[:, :3]  # x, y, z
        rgb = data[:, 3:6] / 255.0  # r, g, b 归一化到[0, 1]
        labels_gt = data[:, 6]  # 标签在第七列

        features = np.hstack((rgb, xyz))

        # 使用 DBSCAN 进行聚类
        dbscan = DBSCAN(eps=5, min_samples=20)
        labels = dbscan.fit_predict(features)

        # 计算 IoU 和精确率、召回率、F1 分数
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)  # 不考虑噪声

        precision_list = []
        recall_list = []
        f1_list = []

        for cluster_label in range(n_clusters):
            true_positive = np.sum((labels_gt == cluster_label) & (labels == cluster_label))
            false_positive = np.sum((labels_gt != cluster_label) & (labels == cluster_label))
            false_negative = np.sum((labels_gt == cluster_label) & (labels != cluster_label))

            # 计算 IoU
            intersection = true_positive
            union = true_positive + false_positive + false_negative

            iou = intersection / union if union > 0 else 0

            # 计算精确率、召回率和 F1 分数
            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

            print(f"Cluster {cluster_label}: IoU = {iou:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}")

        # 为上色后的点云生成颜色映射
        max_label = labels.max() if n_clusters > 0 else 0
        color_map = np.zeros((xyz.shape[0], 3))

        for label in range(max_label + 1):  # 需要处理噪声（-1）情况
            if label in unique_labels:
                color_map[labels == label] = plt.get_cmap("tab20")(label / (max_label + 1))[:3]
            else:
                color_map[labels == -1] = [0, 0, 0]  # 将噪声标记为黑色

        # 创建 Open3D 点云对象并设置颜色
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(color_map)

        # 保存上色后的点云图像
        output_image_path = f"E:/3D-point-clouds-plant-phenotyping/output/colored_point_cloud_dbscan_{i}.png"
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)  # 不显示窗口
        vis.add_geometry(pcd)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        # 设置相机视角
        ctr = vis.get_view_control()
        ctr.set_front([0.5, -0.5, -0.5])  # 调整前视方向
        ctr.set_lookat([0, 0, 0])  # 设置观察点为原点
        ctr.set_up([0, 0, 1])  # 设置上方方向
        ctr.set_zoom(0.5)  # 调整缩放比例

        # 保存图像
        vis.capture_screen_image(output_image_path)
        vis.destroy_window()

        i += 1
