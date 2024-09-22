import os
import sys
import numpy as np
import math
import open3d as o3d
import copy
# import gdbscan
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gdbscan-master')))


UNCLASSIFIED = -2
NOISE = -1

def GDBSCAN(points, n_pred, min_card, w_card):
    cluster_id = 0
    for point in points:
        if point.cluster_id == UNCLASSIFIED:
            if _expand_cluster(points, point, cluster_id, n_pred, min_card,
                               w_card):
                cluster_id = cluster_id + 1
    clusters = {}
    for point in points:
        key = point.cluster_id
        if key in clusters:
            clusters[key].append(point)
        else:
            clusters[key] = [point]
    return list(clusters.itervalues())


def _expand_cluster(points, point, cluster_id, n_pred, min_card, w_card):
    if not _in_selection(w_card, point):
        points.change_cluster_id(point, UNCLASSIFIED)
        return False

    seeds = points.neighborhood(point, n_pred)
    seeds = list(seeds)  # make sure seeds is a list
    print("Initial seeds:", seeds)  # this is for debug 
    if not _core_point(w_card, min_card, seeds):
        points.change_cluster_id(point, NOISE)
        return False

    points.change_cluster_ids(seeds, cluster_id)
    seeds.remove(point)

    while len(seeds) > 0:
        current_point = seeds[0]
        result = points.neighborhood(current_point, n_pred)
        if w_card(result) >= min_card:
            for p in result:
                if w_card([p]) > 0 and p.cluster_id in [UNCLASSIFIED, NOISE]:
                    if p.cluster_id == UNCLASSIFIED:
                        seeds.append(p)
                    points.change_cluster_id(p, cluster_id)
        seeds.remove(current_point)
    return True


def _in_selection(w_card, point):
    return w_card([point]) > 0


def _core_point(w_card, min_card, points):
    return w_card(points) >= min_card


class Points:
    def __init__(self, points):
        self.points = points

    def __iter__(self):
        for point in self.points:
            yield point

    def __repr__(self):
        return str(self.points)

    def get(self, index):
        return self.points[index]

    def neighborhood(self, point, n_pred):
        return filter(lambda x: n_pred(point, x), self.points)

    def change_cluster_ids(self, points, value):
        for point in points:
            self.change_cluster_id(point, value)

    def change_cluster_id(self, point, value):
        index = (self.points).index(point)
        self.points[index].cluster_id = value

    def labels(self):
        return set(map(lambda x: x.cluster_id, self.points))


class Point:
    def __init__(self, x, y, z, r, g, b):
        self.x = x
        self.y = y
        self.z = z
        self.r = r
        self.g = g
        self.b = b
        self.cluster_id = UNCLASSIFIED

    def __repr__(self):
        return '(x:{}, y:{}, z:{}, r:{}, g:{}, b:{}, cluster:{})' \
            .format(self.x, self.y, self.z, self.r, self.g, self.b, self.cluster_id)

def n_pred(p1, p2):
    distance_threshold = 0.5
    color_threshold = 30

    spatial_distance = math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
    color_distance = math.sqrt((p1.r - p2.r)**2 + (p1.g - p2.g)**2 + (p1.b - p2.b)**2)

    return spatial_distance <= distance_threshold and color_distance <= color_threshold

def w_card(points):
   #  print(points)
   #  print(type(points))
    return len(list(points))

def load_data():
    data_dir = 'E:/intern/LAST-Straw/data/'
    all_files = sorted(os.listdir(data_dir))
    scans = [fname for fname in all_files if fname.endswith(".xyz")]
    data = np.loadtxt(data_dir + scans[16], comments='//')  # type: ignore
    return data

def assign_colors(points, num_clusters):
    colors = np.random.rand(num_clusters, 3)
    for point in points:
        if point.cluster_id == NOISE:
            point.color = [0, 0, 0]  # black for noise
        else:
            point.color = colors[point.cluster_id % num_clusters]

points_data = load_data()
points = [Point(data[0], data[1], data[2], data[3], data[4], data[5]) for data in points_data]

min_card = 3
clustered_points = GDBSCAN(Points(points), n_pred, min_card, w_card)

# visualise results
unique_cluster_ids = list(set(point.cluster_id for point in clustered_points))
num_clusters = len(unique_cluster_ids)
assign_colors(clustered_points, num_clusters)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector([(point.x, point.y, point.z) for point in clustered_points])
pcd.colors = o3d.utility.Vector3dVector([point.color for point in clustered_points])

o3d.visualization.draw_geometries([pcd])