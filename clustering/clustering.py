from json import load
import numpy as np
import sys
import os
import open3d as o3d
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from myutils.image_with_bb import get_bounding_boxed_image
import myutils.point_cloud as pc
# import myutils.boundingboxes as bb
from sklearn.cluster import KMeans

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def test_with_Kmean(features, points, config, range_for_K = 7, save_pc=False, index = 0, show_image_bouningbox = True, image=None):
    """Testing clustering the points with Kmeans_alg from sklearn
    ----------
    Parameters
    features: is the information we use to cluster
    points:
    config: the file
    
    output: None
    
    """

    print('[Now create the clustering with Kmeans...]')
    # print(image)
    for i in range(5, range_for_K):
        labels = KMeans(n_clusters=i).fit(features).labels_
        #labels for points in the same cluster

        pc_kmeans = pc.get_point_clouds_with_labels(points, list(labels))
        pc.show_point_cloud(pc_kmeans)
        if save_pc: pc.save_point_cloud_with_labels((255*points).astype(np.int32), list(labels), config, i, index)
        if show_image_bouningbox:
            pc_kmeans = pc.reduce_point_cloud(pc_kmeans, max_fraction=0.8)
            get_bounding_boxed_image(image, pc_kmeans)


def get_Kmeans_labels(K, eigenvector):
    print('[Now create the clustering with Kmeans...]')
    r_eig = eigenvector.reshape(-1,1)
    labels = KMeans(n_clusters=K).fit(r_eig).labels_
        #eigenvector is 1D, should be 2d?    
    print(labels)
    return labels



   
