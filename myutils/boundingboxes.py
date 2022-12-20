import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import myutils.point_cloud as pc
from myutils.parse import parse_args, get_cfg_node
import open3d as o3d
from clustering.DBSCAN_clustering import test_with_DBSCAN_by_monocolor_pc, get_pc_DBSCAN

import torch


def get_bounding_box(point_cloud, color = [1,0,0]):
    bb = point_cloud.get_oriented_bounding_box()
    bb.color = color
    return bb

def get_lineset_for_a_bounding_box(bb):
    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(bb)
    return line_set


def show_bounding_box(point_cloud, color = [1,0,0]):
    bb = point_cloud.get_oriented_bounding_box()
    bb.color = color
    o3d.visualization.draw_geometries([point_cloud, bb], width =1400, height= 1000)

def bounding_box_all_point_cloud(point_cloud):
    print('[bounding boxe every body...]')
    
    pc_list = pc.get_point_clouds_separeted_by_color(point_cloud)
    colors = pc.get_point_cloud_colors(point_cloud)

    for color in colors:
        mono_pc = pc.get_point_cloud_by_color(point_cloud, color)
        bb = get_bounding_box(mono_pc)
        pc_list.append(bb)

    pc.show_point_clouds(pc_list)
    return pc_list

def get_one_bounding_box_by_color(point_cloud_complete, point_cloud_color):
    ply = pc.get_point_cloud_by_color(point_cloud_complete, point_cloud_color)
    bb = get_bounding_box(ply)
    return bb

def get_each_bounding_box(point_cloud):

    colors = pc.get_point_cloud_colors(point_cloud)
    for color in colors:
        bb = get_one_bounding_box_by_color(point_cloud, color)
        pc.show_point_clouds([point_cloud, bb])

def get_each_bounding_boxes_with_DBSCAN(point_cloud_Kmeans):
    """receives a point color after K means at its original, can be easily adptadet for the role pipeline
    separetes the point cloud in each color and then show each bounding box
    
    """
    # colors = pc.get_point_cloud_colors(point_cloud_Kmeans)
    print('[bounding box with DBSCAN...]')
    # pc.show_point_cloud(point_cloud_Kmeans)
    monocolor_point_clouds = pc.get_point_clouds_separeted_by_color(point_cloud_Kmeans)
    bb = []
    temp = []
    bb.append(point_cloud_Kmeans)
    
    for monocolor_point_cloud in monocolor_point_clouds:
        # pc.show_point_cloud(monocolor_point_cloud)
        pc_dbscaned = test_with_DBSCAN_by_monocolor_pc(monocolor_point_cloud)
        # pc.show_point_cloud(pc_dbscaned)
        monocolors_pc_dbscaneds = pc.get_point_clouds_separeted_by_color(pc_dbscaned)
        for pcm in monocolors_pc_dbscaneds:
            # pc.show_point_cloud(pcm)
            points = np.array(pcm.points)
            N = points.shape[0]
            # print(points.shape[0])
            if N>400:
                bbm = get_bounding_box(pcm) #pronblem here is the points are to close
                bb.append(bbm)
                temp.append(point_cloud_Kmeans)
                temp.append(bbm)
                pc.show_point_clouds(temp)
                temp = []
                # print(bb)
        
    # pc_list = []

    # pc_list.append(point_cloud_default)
    pc.show_point_clouds(bb)

def get_all_bounding_boxes_with_DBSCAN(point_cloud_Kmeans, min_samples = 400, save_pc = False):
    """given a point cloud with Kmeans - separate it with dbscan and then get all its bounding boxes
    
    returns a list of bounding boxes
    """
    pcs_dbscan = get_pc_DBSCAN(point_cloud_Kmeans)
    complete_list = []
    bb_list = []
    
    for pc_dbscan in pcs_dbscan:
        complete_list.append(pc_dbscan)
        monocolors_pc_dbscan = pc.get_point_clouds_separeted_by_color(pc_dbscan)
        for pcm in monocolors_pc_dbscan:
            points = np.array(pcm.points)
            N = points.shape[0]
            if N>min_samples:# and N<8000:
                bbm = get_bounding_box(pcm) #pronblem here is the points are to close
                complete_list.append(bbm)
                bb_list.append(bbm)
               
    
    # pc.show_point_clouds(bb_list)
    pc.show_point_clouds(complete_list)
    if save_pc: pc.save_point_cloud(bb_list)

    return bb_list

def get_coordinates_of_bb_vertices(bounding_box):
    """return the bounding box point as a np array"""
    bb_points = np.array(bounding_box.get_box_points())
    return bb_points
    



if __name__ == "__main__":

    pck6 = pc.load_point_cloud('temp_pc\conferenceRoom_1_6mean_1imagesUsed_2022-11-29-11-54_point_cloud.ply')


    # pck6_dbscan = pc.load_point_cloud('temp_pc\DBSCANconferenceRoom_1_6mean_1imagesUsed_2022-11-29-11-54_point_cloud.ply')
    # pc.show_point_cloud(pck6)

    # pc.show_point_cloud(monocolor_lamps)
    get_all_bounding_boxes_with_DBSCAN(pck6)

    # pc.show_point_cloud(monocolor)



