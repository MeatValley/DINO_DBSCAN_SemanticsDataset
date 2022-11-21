import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import myutils.point_cloud as pc
from myutils.parse import parse_args, get_cfg_node
import open3d as o3d
from clustering.clustering import test_with_DBSCAN, test_with_DBSCAN_by_pc



def get_bounding_box(point_cloud, color = [1,0,0]):
    bb = point_cloud.get_oriented_bounding_box()
    bb.color = color
    return bb

def show_bounding_box(point_cloud, color = [1,0,0]):
    bb = point_cloud.get_oriented_bounding_box()
    bb.color = color
    o3d.visualization.draw_geometries([point_cloud, bb], width =1400, height= 1000)

def bounding_box_all_point_cloud(point_cloud):
    
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

def reunion1_show_by_color():
    pck6 = pc.load_point_cloud('temp_pc\conferenceRoom1_6mean_1imagesUsed_2022-11-16-09-51_point_cloud.ply')
    pc.show_point_cloud(pck6)
    pc.show_point_clouds_separeted_by_color(pck6, save_pc=True)
    
def reunion2_dbscan():
    pc1 = test_with_DBSCAN('temp_pc\pc_separeted_by_color_color1.ply') #eps 10, min = 10
    pc.show_point_cloud(pc1)
    # test_with_DBSCAN('temp_pc\pc_separeted_by_color_color3.ply') 
    # test_with_DBSCAN('temp_pc\pc_separeted_by_color_color2.ply')
    # test_with_DBSCAN('temp_pc\pc_separeted_by_color_color4.ply')
    # test_with_DBSCAN('temp_pc\pc_separeted_by_color_color5.ply')

def bounding_boxes_with_DBSCAN(point_cloud_Kmeans, point_cloud_default):
    """receives a point color after K means at its original, can be easily adptadet for the role pipeline"""
    # colors = pc.get_point_cloud_colors(point_cloud_Kmeans)
    print('[bounding box with DBSCAN...]')
    # reduced_pc = pc.reduce_point_cloud(point_cloud_Kmeans, max_points=20) #problema nao eh reduzindo
    monocolor_point_clouds = pc.get_point_clouds_separeted_by_color(point_cloud_Kmeans)
    pc.show_point_clouds(monocolor_point_clouds)
    print('a')
    for point_cloud in monocolor_point_clouds:
        print('x')
        pc_dbscaned = test_with_DBSCAN_by_pc(point_cloud)
        pc.show_point_cloud(pc_dbscaned)




if __name__ == "__main__":
    # pck3 = pc.load_point_cloud('temp_pc\conferenceRoom1_3mean_1imagesUsed_2022-11-16-09-37_point_cloud.ply') #good
    # pck4 = pc.load_point_cloud('temp_pc\conferenceRoom1_4mean_1imagesUsed_2022-11-16-09-37_point_cloud.ply')
    pck6 = pc.load_point_cloud('temp_pc\conferenceRoom1_6mean_1imagesUsed_2022-11-16-09-51_point_cloud.ply')
    # pck7 = pc.load_point_cloud('temp_pc\conferenceRoom1_7mean_1imagesUsed_2022-11-16-09-51_point_cloud.ply')
    default = pc.load_point_cloud('temp_pc/1image_default.ply')
    # pc.show_point_cloud(pck6)
    bounding_box_all_point_cloud(pck6)

    # k9 = pc.load_point_cloud('configs\logs\point_clouds\s0279_0010mean0_2022-10-14-14-50_point_cloud_colors.ply')

    # reunion1_show_by_color()
    # reunion2_dbscan()
    # bounding_boxes_with_DBSCAN(pck6, default)
    


