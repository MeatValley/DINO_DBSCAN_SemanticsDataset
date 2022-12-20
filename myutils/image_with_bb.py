import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from myutils.image import load_image, plot
from myutils.boundingboxes import get_all_bounding_boxes_with_DBSCAN
import myutils.point_cloud as pc
import numpy as np
import matplotlib.pyplot as plt

from geometry.camera import Camera
from dataset.semantics_dataset import SemanticsDataset
import myutils.boundingboxes as bb
import myutils.image_with_bb_utils as ut



def get_bounding_boxed_image(image_path, point_cloud_Kmeans):
    image = load_image(image_path)
    default = load_image(image_path)
    dims_image = tuple(np.asarray(image).shape[0:2][::-1])
    # plot(image, 'original image')
    path = image_path.split('\\')
    core = path[-1]
    room_list = core.split('_')
    room_name = room_list[2]+'_'+room_list[3]
    room_path = ''
    point_cloud_name = room_name + '.txt'
    for part in path:
        room_path = os.path.join(room_path, part)
        if part == room_name: break
    dataset = SemanticsDataset(room_path, point_cloud_name)

    pose_path = dataset.get_pose_path(image_path)
    depth_path = dataset.get_depth_path(image_path)
    cam_f = dataset.get_pose(pose_path)
    cam = Camera(K=cam_f.K, dimensions=dims_image, Tcw=np.linalg.inv(cam_f.rotation_matrix))
    line_image = image_with_bb(image, point_cloud_Kmeans, cam, image)
    ut.show_comparation(default, line_image)

def image_with_bb(room_image, point_cloud_Kmeans, camera, default):
    """receives a image, the Kmeans segmenteded for the image a the camera for the image
    
    returns the image with the bounded boxes
    """
    # plot(room_image)
    # default = backup(room_image)
    
    bb_list = get_all_bounding_boxes_with_DBSCAN(point_cloud_Kmeans, min_samples=400)
    point_coordinates = []

    for bbm in bb_list:
        bb_vertices = bb.get_coordinates_of_bb_vertices(bbm)
        # print(bb_vertices) #list of 8x3 position array that represents a bb
        for point in bb_vertices:
            # print(point)
            point_coordinates.append(point)

    point_coordinates = np.array(point_coordinates)
    point_coordinates = point_coordinates/255 #clustreing.clustering line 34
    pixel_coordinates = camera.project_for_bb(point_coordinates)


    bbs_vertices_pixel_list = []
    bb_vertices_pixel_list = []
    for n,pixel in enumerate(pixel_coordinates):
        x, y = int((pixel[1]) * camera.H), int((pixel[0]) * camera.W)
        v = (y,x)
        bb_vertices_pixel_list.append(v)
        if len(bb_vertices_pixel_list) == 8: 
            bbs_vertices_pixel_list.append(bb_vertices_pixel_list)
            bb_vertices_pixel_list = []

            

        #x,y is the vertices of a 3d bb in the image

    for bbvp in bbs_vertices_pixel_list:
        print(bbvp)
        line_image = ut.get_bb_draw(bbvp, room_image)
    
    # plot(line_image, 'final image')
    # plot(default, 'def')


    # plot(line_image)

    # print(bb_list)
    return line_image

def get_each_bb_for_image(image_path, point_cloud_Kmeans):
    image = load_image(image_path)
    default = load_image(image_path)
    dims_image = tuple(np.asarray(image).shape[0:2][::-1])
    plot(image, 'original image')
    path = image_path.split('\\')
    core = path[-1]
    room_list = core.split('_')
    room_name = room_list[2]+'_'+room_list[3]
    room_path = ''
    point_cloud_name = room_name + '.txt'
    for part in path:
        room_path = os.path.join(room_path, part)
        if part == room_name: break
    dataset = SemanticsDataset(room_path, point_cloud_name)

    pose_path = dataset.get_pose_path(image_path)
    depth_path = dataset.get_depth_path(image_path)
    cam_f = dataset.get_pose(pose_path)
    cam = Camera(K=cam_f.K, dimensions=dims_image, Tcw=np.linalg.inv(cam_f.rotation_matrix))

    monocolor_pointclouds = pc.get_point_clouds_separeted_by_color(point_cloud_Kmeans)
    for monoclor_pointcloud in monocolor_pointclouds:
        line_image = image_with_bb(image, monoclor_pointcloud, cam, image)
        ut.show_comparation(default, line_image)

def reunionv31():
    pck6 = pc.load_point_cloud('temp_pc\conferenceRoom1_6mean_1imagesUsed_2022-11-16-09-51_point_cloud.ply')
    monocolor_lamps = pc.load_point_cloud('temp_pc\separated by color\pc_separeted_by_color_color1.ply')
    monocolor_board = pc.load_point_cloud('temp_pc\separated by color\pc_separeted_by_color_color3.ply')
    monocolor_chair = pc.load_point_cloud('temp_pc\separated by color\pc_separeted_by_color_color4.ply')
    monocolor_table = pc.load_point_cloud('temp_pc\separated by color\pc_separeted_by_color_color5.ply') ### problem(?)
    monocolor_roof = pc.load_point_cloud('temp_pc\separated by color\pc_separeted_by_color_color2.ply')
    path = 'database_organized\database_organized_Area1\conferenceRoom_1\color\camera_0d600f92f8d14e288ddc590c32584a5a_conferenceRoom_1_frame_0_domain_rgb.png'
    default = pc.load_point_cloud('temp_pc/1image_default.ply')
    get_bounding_boxed_image(path, monocolor_table)

def reunionv32():
    
    pck6 = pc.load_point_cloud('temp_pc\conferenceRoom_1_6mean_1imagesUsed_2022-11-29-16-22_point_cloud.ply')

    # pc.show_point_cloud(pck6)
    # get_bounding_boxed_image(path, pck6)
    get_each_bb_for_image(path, pck6)

if __name__ == "__main__":
    pck6 = pc.load_point_cloud('temp_pc\conferenceRoom_1_6mean_1imagesUsed_2022-11-29-18-21_point_cloud.ply')
    
    monocolor0 = pc.load_point_cloud('temp_pc\A_monocolor0.ply')
    monocolor1 = pc.load_point_cloud('temp_pc\A_monocolor1.ply')
    monocolor2 = pc.load_point_cloud('temp_pc\A_monocolor2.ply')
    monocolor3 = pc.load_point_cloud('temp_pc\A_monocolor3.ply')
    monocolor4 = pc.load_point_cloud('temp_pc\A_monocolor4.ply')

    path = 'database_organized\database_organized_Area1\conferenceRoom_1\color\camera_042a479869b44a7c9159922f19a285ea_conferenceRoom_1_frame_37_domain_rgb.png'
    default = pc.load_point_cloud('temp_pc/1image_default.ply')


    # img = load_image(path)
    # plot(img)
    # reunionv32()
    pc.show_point_cloud(pck6)
    get_bounding_boxed_image(path, pck6)




    # get_bounding_boxed_image(path, monocolor_table)
    # get_each_bb_for_image(path, pck6)

# get_bounding_boxed_image('database_organized\database_organized_Area1\conferenceRoom_1\color\camera_0d600f92f8d14e288ddc590c32584a5a_conferenceRoom_1_frame_0_domain_rgb.png')