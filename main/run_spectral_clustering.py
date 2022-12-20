import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
import myutils.point_cloud as pc
import myutils.parse as parse
from dataset.semantics_dataset import SemanticsDataset
import matplotlib.pyplot as plt
from feature_extractor.DINO_utils import get_feature_dictonary, intra_distance
import open3d as o3d
from clustering.clustering import get_Kmeans_labels
from myutils.image import plot
from clustering.spectral_clustering import get_spectral_clustering
import random




with torch.no_grad():
    def run_spectral_clustering(file, number_images = 1, save_pc=False, K=12, robust_mean = True, max_pooling = False, run_complete = False, show2d = False):
        """ Runs DINO features code adapted for SemanticsDataset with
        spectral clustering optimized
        
        """
        print(f'[running "run": reading {number_images} images from dataset, and Kmeans with K in range [2,{K}] ...]')
        config = parse.get_cfg_node(file)

        dataset = SemanticsDataset(config.data.path, config.data.point_cloud_name)
        # print(config.data.path,config.data.point_cloud_name)  
        dictionary = {}
        ind = number_images

        print("[getting a sample: ]")
        for i, sample in enumerate(iter(dataset)): #enumerate is just for i to be a counter
            
            if (i == ind) and (not run_complete):
                break

            print('[we are in a sample...]')
            for j in range(len(sample["correspondances"])):
                dictionary, patches= get_feature_dictonary(dictionary, sample["correspondances"][j], sample["image_DINO_features"][j], dataset) 
                # plot_2dimages_for_pc_im_correspondances_matching(sample, j) 

            dictionary = reduce_dictionary(dictionary, max_points=10000)
            mapa, eigenvector = get_spectral_clustering(dictionary)
            
            for i in range(2, K):
                labels = get_Kmeans_labels(i, eigenvector)
                #labels is [0 0 0 1 3 0 1 ...] shape  = number of points
                corresp = {}
                for n,key in enumerate(mapa): 
                    corresp[mapa[key]] = labels[key]

                pc_spec = pc.get_point_cloud_spectral_clustering(corresp, dataset.point_cloud_points)

                pc.show_point_cloud(pc_spec)
                name = 'spec_clutering'+str(i)+'means'+'beta03_while01.ply'
                path = os.path.join('temp_pc', name)

                if save_pc: pc.save_point_cloud(pc_spec, path)


            # pc.show_point_cloud_spectral_clustering(corresp, dataset.point_cloud_points)




 
    
def plot_2dimages_for_pc_im_correspondances_matching(sample, j):
    print('[trying to plot the debugging matches...]')
    plt.figure()
    plt.subplot(1, 5, 1)
    plt.imshow(sample["image"][j])
    plt.subplot(1, 5, 2)
    plt.imshow(sample["proj3dto2d_depth"][j])
    plt.subplot(1, 5, 3)
    plt.imshow(sample["image"][j]-sample["proj3dto2d"][j])
    plt.subplot(1, 5, 4)
    plt.imshow(sample["proj3dto2d"][j])  
    plt.subplot(1, 5, 5)
    plt.imshow(sample["depth_map"][j])
    plt.show()

def reduce_dictionary(dictionary, max_points=10000):
    N = len(dictionary)
    # print(N)70000 e cacetada
    reduced_dictionary = {}
    if N > max_points:
        for i in range(max_points):
            point, feature = random.choice(list(dictionary.items()))
            reduced_dictionary[point] = feature
            

    # print(reduced_dictionary)
    
    return reduced_dictionary

def save_default_point_cloud(points, colors):
    print('[saving default pc...]')
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    pc.save_point_cloud(point_cloud, 'temp_pc/default.ply')

def get_max_pooling(dictionary_features):
    max_pooling_mean = np.zeros(384)
    for i in range(384):
        for k in range(len(dictionary_features)): #okay
            if k == 0: max_temp = dictionary_features[k][i]
            else:
                if max_temp<dictionary_features[k][i]: max_temp =dictionary_features[k][i]
        max_pooling_mean[i] = max_temp

    dino = max_pooling_mean
    return dino