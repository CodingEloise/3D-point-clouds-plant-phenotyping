import os
import open3d as o3d

data_dir = 'E:\\3D-point-clouds-plant-phenotyping\\output\\labeled_data'
all_files = sorted(os.listdir(data_dir))
pcd_files = [fname for fname in all_files if fname.endswith(".ply")]

def get_point_cloud_info(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    points = len(pcd.points)
    
    base_name = os.path.basename(file_path)
    parts = base_name.split('_')
    if len(parts) >= 4 and '-' in parts[3]:
        group_part = parts[3]
        group_name = group_part.split('-')[0] 
        if not group_name.isdigit():  
            return None, None, None
    else:
        return None, None, None  
    return group_name, points, pcd

def find_largest_point_cloud_by_group(point_cloud_files, data_dir):
    max_point_clouds_by_group = {}

    for file_name in point_cloud_files:
        file_path = os.path.join(data_dir, file_name)
        group_name, points, pcd = get_point_cloud_info(file_path)
        
        if group_name is None:
            continue  
        
        group_index = int(group_name)
        if group_index not in max_point_clouds_by_group or points > max_point_clouds_by_group[group_index]['points']:
            max_point_clouds_by_group[group_index] = {'name': file_name, 'points': points, 'pcd': pcd}

    max_index = max(max_point_clouds_by_group.keys()) if max_point_clouds_by_group else -1
    largest_point_clouds = [None] * (max_index + 1)

    for group_index, point_cloud in max_point_clouds_by_group.items():
        largest_point_clouds[group_index] = point_cloud['name']
        print(f"Group: {group_index}, Largest Point Cloud: {point_cloud['name']}")
      #   o3d.visualization.draw_geometries([point_cloud['pcd']], window_name=f"Group {group_index}")

    return largest_point_clouds

largest_point_clouds = find_largest_point_cloud_by_group(pcd_files, data_dir)
print("Largest Point Clouds Array:", largest_point_clouds)
