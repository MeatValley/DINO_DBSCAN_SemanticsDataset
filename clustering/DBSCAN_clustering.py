from json import load
import numpy as np
import sys
import os
import open3d as o3d
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import myutils.point_cloud as pc

from sklearn.cluster import DBSCAN



def test_with_DBSCAN_by_monocolor_pc(point_cloud):
    """given a monocolor point_cloud it separates by Kmeans"""
    print('[running DBSCAN receiving a point cloud...]')

    points = point_cloud.points
    points = np.array(points)
    clustering = DBSCAN(eps=10, min_samples=30).fit(points)
    labels = clustering.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('number of clusters ', n_clusters_)
    ret_pc = pc.get_point_clouds_with_labels(points, list(labels), random_colors=True)
    return ret_pc

def get_pc_DBSCAN(point_cloud, show_pc_dbscanned = False):
    monocolor_point_clouds = pc.get_point_clouds_separeted_by_color(point_cloud)
    pc_dbscanned = []
    points = []
    colors = []
    point_cloud_dbscan = o3d.geometry.PointCloud()

    for monocolor_point_cloud in monocolor_point_clouds:
        pc_dbscan = test_with_DBSCAN_by_monocolor_pc(monocolor_point_cloud)
        pc_dbscanned.append(pc_dbscan)
    #     points.append(pc_dbscan.points)
    #     colors.append(pc_dbscan.colors)    
    # points = np.array(points)
    # colors = np.array(colors)


    # point_cloud_dbscan.points = o3d.utility.Vector3dVector(points)
    # point_cloud_dbscan.colors = o3d.utility.Vector3dVector(colors)


    if show_pc_dbscanned: pc.show_point_clouds(pc_dbscanned)
    return pc_dbscanned


