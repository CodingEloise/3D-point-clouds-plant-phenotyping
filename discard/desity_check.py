import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def compute_density_by_radius(points, radius=0.1):
    pcd_tree = o3d.geometry.KDTreeFlann(points)
    densities = []
    
    for point in points.points:
        [_, idx, _] = pcd_tree.search_radius_vector_3d(point, radius)
        density = len(idx)  # 邻居点的数量
        densities.append(density)
    
    return np.array(densities)

# 加载点云
pcd = o3d.io.read_point_cloud("E:/3D-point-clouds-plant-phenotyping/output/original_data/point_cloud16.ply")
points = np.asarray(pcd.points)

print(len(points))

# 计算密度分布
densities = compute_density_by_radius(pcd, radius=1)

# 你可以进一步分析或可视化密度分布
print("Density distribution:", densities)


def visualize_density(pcd, densities):
    # Normalize densities for color mapping
    normalized_densities = (densities - np.min(densities)) / (np.max(densities) - np.min(densities))
    colormap = plt.get_cmap("jet")
    colors = colormap(normalized_densities)[:, :3]
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

# 可视化密度
visualize_density(pcd, densities)
