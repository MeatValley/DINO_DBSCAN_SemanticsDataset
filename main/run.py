import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
import myutils.point_cloud as pc
import myutils.parse as parse
from dataset.semantics_dataset import SemanticsDataset
import matplotlib.pyplot as plt
import cv2
from feature_extractor.DINO_utils import get_feature_dictonary, intra_distance
import open3d as o3d
from clustering.clustering import test_with_Kmean
from myutils.image import plot

with torch.no_grad():
    def run(file, number_images = 1, save_pc=False, K=12, max_pooling = False):
        """ Runs DINO features code adapted for SemanticsDataset
        
        """
        print(f'[running "run": reading {number_images} images from dataset, and Kmeans with K in range [2,{K}] ...]')
        config = parse.get_cfg_node(file)

        dataset = SemanticsDataset(config.data.path, config.data.point_cloud_name, stop_image=number_images)
        dictionary = {}
        ind = number_images

        print("[getting a sample: ]")
        for i, sample in enumerate(iter(dataset)): #enumerate is just for i to be a counter

            print('[we are in a sample...]')

            for j in range(len(sample["correspondances"])):
                dictionary, patches= get_feature_dictonary(dictionary, sample["correspondances"][j], sample["image_DINO_features"][j], dataset) 
                # plot_2dimages_for_pc_im_correspondances_matching(sample, j) 

            # print("[calculating Intra distance...]")
            # dist = intra_distance(dictionary, perturb=False) ########### tsne is too complicate bc is to much points (110.000 here and 30.000 scannet)

            print('[doing the mean of the features...]')
            
            dataset.point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            keys_sel = {}
            raw_features = []
            finale_features = []
            points = []
            colors = []
            normals = []
            for j, point in enumerate(dataset.point_cloud_points): # for each point
                
                if j in dictionary.keys(): #dictionary is just for one part


                    ################################################################### - robust mean
                    mean = sum(dictionary[j])/len(dictionary[j]) #mean of features
                    raw_features.append(mean)
                    #for each point we can have more than 1 feature stacked in dictionary, so make a mean

                    feat_temp = dictionary[j]
                    feat_temp_dist=[np.linalg.norm(robust_feature - mean) for robust_feature in dictionary[j]]
                    #distance of each feature to the mean

                    n_new_feat = max([int(0.6*len(feat_temp)), 1])
                    # we catch just the ones near
                    feat_temp_filtr = sorted(zip(feat_temp, feat_temp_dist), key= lambda x: x[1])[0:n_new_feat]
                    feat_temp_filtr = [x[0] for x in feat_temp_filtr]                    # just add the ones near

                    robust_mean = sum(feat_temp_filtr)/len(feat_temp_filtr)

                    #new mean (robust mean)
                    keys_sel[len(points)]=j
                    dino = robust_mean
                    ###########################################################################
                    if max_pooling:
                        max_pooling_mean = np.zeros(384)
                        for i in range(384):
                            for k in range(len(dictionary[j])): #okay
                                if k == 0: max_temp = dictionary[j][k][i]
                                else:
                                    if max_temp<dictionary[j][k][i]: max_temp = dictionary[j][k][i]
                            max_pooling_mean[i] = max_temp

                        dino = max_pooling_mean



                    point = np.asarray(point)
                    normal = np.asarray(dataset.point_cloud.normals[j])
                    color = dataset.point_cloud_colors[j]/255
                            
                    # dino = mean
                    # dino = robust_mean
                    
                    points.append(point[None, :]) # [None, :] treats point as a single element
                    # x -> [x]
                    colors.append(color[None, :])
                    normals.append(normal[None, :])
                    #creates the finale feature

                    feature  = dino[None, :]
                    # feature  = color[None, :]
                    finale_features.append(feature)

            # plot(sample["image"])
            # visualization_of_pc(points, colors)
            # save_default_point_cloud(points, colors)
            # print('filane_features so far...', finale_features)

            finale_features = np.concatenate(finale_features, axis=0)
            # print('finale features: ',finale_features.shape)
            points = np.concatenate(points, axis=0) #axis = 0 treats each independent
            colors = np.concatenate(colors, axis=0)

            # finale_features = (finale_features - finale_features.mean(axis=0))/finale_features.std(axis=0)
            
            test_with_Kmean(finale_features, points, config, range_for_K=K, save_pc= save_pc, index = number_images)
            if i == ind-1: 
                print('[breaking....]')
                return



        test_with_Kmean(finale_features, points, config, range_for_K=K, save_pc= save_pc, index = number_images)
        # test_with_Kmean(raw_features, points, config, range_for_K=4)
    
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

def visualization_of_pc(points, colors):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
            
        print('[vizualiation of pc...]')
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(width = 1300, height = 700)
        vis.add_geometry(point_cloud)
        vis.run()  # user picks points
        vis.destroy_window()
        return point_cloud

def save_default_point_cloud(points, colors):
    print('[saving default pc...]')
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    pc.save_point_cloud(point_cloud, 'temp_pc/default.ply')