from json import load
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import myutils.point_cloud as pc
from sklearn.cluster import KMeans

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def test_with_Kmean(features, points, config, range_for_K = 7, save_pc=False, index = 0):
    """Testing clustering the points with Kmeans_alg from sklearn
    ----------
    Parameters
    features: is the information we use to cluster
    points:
    config: the file
    
    output: None
    
    """

    print('[Now create the clustering with Kmeans...]')
    for i in range(2, range_for_K):
        labels = KMeans(n_clusters=i).fit(features).labels_
        #labels for points in the same cluster

        # print(labels)
            
        pc.show_point_clouds_with_labels(points, list(labels))
        if save_pc: pc.save_point_cloud_with_labels((255*points).astype(np.int32), list(labels), config, i, index)
        # print('Clustering')

def test_with_DBSCAN(path):
    print('[running DBSCAN receiving a path...]')
    point_cloud_loaded = pc.load_point_cloud(path)
    pc.show_point_cloud(point_cloud_loaded)
    points = point_cloud_loaded.points
    points = np.array(points)
    clustering = DBSCAN(eps=10, min_samples=30).fit(points)
    labels = clustering.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('number of clusters ', n_clusters_)
    ret_pc = pc.show_point_clouds_with_labels(points, list(labels), random_colors=True)
    return ret_pc

def test_with_DBSCAN_by_pc(point_cloud):
    print('[running DBSCAN receiving a point cloud...]')

    points = point_cloud.points
    points = np.array(points)
    clustering = DBSCAN(eps=10, min_samples=30).fit(points)
    labels = clustering.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('number of clusters ', n_clusters_)
    ret_pc = pc.show_point_clouds_with_labels(points, list(labels), random_colors=True)
    return ret_pc

# print('x')