import os
import shutil
import numpy as np
import sys
import math

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import myutils.point_cloud as pc


def creates_area_folders(destination_path):
    """
    First thing to run: it creates the folders to put the data after download
    - simple function: cretes 6 folders database_Areai (i=1,...,6) with folders 2d and 3d in it

    """

    for i in range(1,7):
        name = 'database_Area' + str(i)
        organized_path = os.path.join(destination_path,'database',name)
        # print(organized_path)
        os.mkdir(organized_path)

    dir_list =[f for f in os.listdir(destination_path) if 'Area' in f] 

    for dir in dir_list:
        name = '2d'
        organized_path = os.path.join(destination_path, dir, name)
        os.mkdir(organized_path)
        name = '3d'
        organized_path = os.path.join(destination_path, dir, name)
        os.mkdir(organized_path)

def creates_organized_area_folders(destination_path):
    for i in range(1,7):
        name = 'database_organized_Area' + str(i)
        organized_path = os.path.join(destination_path, 'database_organized', name)
        # print(organized_path)
        os.mkdir(organized_path)

def create_room_folders(database_path, destination_path):
    """
    receive the path to the database folder->
    path = 'database\database_Area1'
    destination_path = 'database_organized\database_organized_Area1'
    
    returns a Area folder with all rooms folder
    
    """
    path = os.path.join(database_path, '3d')
    dir_list = os.listdir(path)
    try:
        for room in dir_list:
            # print(room)
            organized_path = os.path.join(destination_path, room)
            os.mkdir(organized_path)
    except OSError as error:
        print('#### ERROR: The Room Directory already exists')

def populate_folders_with_point_cloud(database_path, destination_path):
    """populate each room_fodler in the database_organized with its pointcloud"""

    original_path = os.path.join(database_path, '3d')


    dir_list = os.listdir(original_path)
    for room in dir_list:
        if room != '.DS_Store' and not room.endswith('.txt'): 
            room_name = str(room)
            point_cloud_name = room_name+'.txt'
            src_path = os.path.join(original_path, room, point_cloud_name )
            dst_path = os.path.join(destination_path, room)
            print('src', src_path)
            print('dst', dst_path)
            # shutil.copy(src_path, dst_path) 


def populate_folders_with_not_alig_point_cloud(path, area_number, destination_path):
    """populate each room_fodler in the database_organized with its pointcloud"""

    name= 'Area_'+str(area_number)
    original_path = os.path.join(path, name )
    dir_list = os.listdir(original_path)
    for room in dir_list:
        if room != '.DS_Store' and not room.endswith('.txt'):
            room_txt = str(room)
            room_name = room_txt.split('.')[0] 
            room_txt = room_name+'.txt'
            room_txt_rot = room_name+'_not_alig.txt'
            # print(room_name)
            organize_data_name = 'database_organized_Area'+str(area_number)
            print(organize_data_name)
            src_path = os.path.join(original_path, room, room_txt)
            dst_path = os.path.join(destination_path, organize_data_name, room_name, room_txt_rot)
            # print(src_path)
            print(dst_path)
            shutil.copy(src_path, dst_path) 


def populate_folders_with_point_cloud_alig_not_alig(database_path, area_number, destination_path):
    """populate each room_fodler in the database_organized with its pointcloud"""

    area_name = 'Area_'+str(area_number)
    original_path = os.path.join(database_path, area_name)

    dir_list = os.listdir(original_path)
    for room in dir_list:
        if room != '.DS_Store' and not room.endswith('.txt'): 
            room_name = str(room)
            point_cloud_name = room_name+'.txt'
            room_NA = room+'_aligned.txt'
            # room_NA = room+'_not_aligned.txt'
            src_path = os.path.join(original_path, room, point_cloud_name )
            dst_path = os.path.join(destination_path, room, room_NA)
            print('src', src_path)
            print('dst', dst_path)
            shutil.copy(src_path, dst_path) 



def create_2dfolders(destination_path):
    """
    receive the path to the database folder->
    
    creates a rgb, pose and depth folder for each room
    
    """
    dir_list = os.listdir(destination_path)
    try:
        for room in dir_list:
            # print(room)
            c_organized_path = os.path.join(destination_path, room, 'color')
            p_organized_path = os.path.join(destination_path, room, 'pose')
            d_organized_path = os.path.join(destination_path, room, 'depth')
            
            os.mkdir(p_organized_path)
            os.mkdir(d_organized_path)
            os.mkdir(c_organized_path)
    except OSError as error:
        print('#### ERROR: The 2d folders already exists')

def popuate_folders_with_its_pose_files(database_path, destination_path):
    """
    receive the path to the database folder->
    
    populate the pose folder of each room in the organized database with 
    the pose files of that room
    """
    original_path = os.path.join(database_path, '2d')
    original_path = os.path.join(original_path, 'pose')
    dir_list = os.listdir(original_path)

    for uid in dir_list:
        if uid != '.gitkeep':
            split = list(uid.split('_'))
            room = str(split[2])+'_'+ str(split[3])
            # print(split)
            src_path = os.path.join(original_path, uid )
            dst_path = os.path.join(destination_path, room, 'pose')
            shutil.copy(src_path, dst_path)       
                
def popuate_folders_with_its_color_images(database_path, destination_path):
    """
    receive the path to the database folder->
    
    populate the color folder for each room camera_874b1dfd225c45dd9fc79b1414c44ca5_conferenceRoom_1_frame_54_domain_rgb
    
    """
    original_path = os.path.join(database_path, '2d')
    organized_path = destination_path
    original_path = os.path.join(original_path, 'rgb')
    room_list = os.listdir(organized_path)
    for room in room_list:
        json_path = os.path.join(organized_path, room, 'pose')
        json_list = os.listdir(json_path)
        for json_file in json_list:
            name_list = json_file.split('_')
            rgb_image_name = name_list[0]+'_'+ name_list[1]+'_'+name_list[2]+'_'+name_list[3]
            rgb_image_name+='_'+name_list[4]+'_'+name_list[5]+'_'+name_list[6]+'_rgb.png'

            src_path = os.path.join(original_path, rgb_image_name )
            dst_path = os.path.join(destination_path, room, 'color')
            shutil.copy(src_path, dst_path) 
           
def popuate_folders_with_its_depth_images(database_path, destination_path):
    """
    receive the path to the database folder->
    
    populate the color folder for each room camera_874b1dfd225c45dd9fc79b1414c44ca5_conferenceRoom_1_frame_54_domain_rgb
    
    """
    original_path = os.path.join(database_path, '2d')
    organized_path = destination_path
    original_path = os.path.join(original_path, 'depth')
    room_list = os.listdir(organized_path)
    for room in room_list:
        json_path = os.path.join(organized_path, room, 'pose')
        json_list = os.listdir(json_path)
        for json_file in json_list:
            name_list = json_file.split('_')
            depth_image_name = name_list[0]+'_'+ name_list[1]+'_'+name_list[2]+'_'+name_list[3]
            depth_image_name+='_'+name_list[4]+'_'+name_list[5]+'_'+name_list[6]+'_depth.png'

            src_path = os.path.join(original_path, depth_image_name )
            dst_path = os.path.join(destination_path, room, 'depth')
            shutil.copy(src_path, dst_path) 

def rotate_the_point_cloud_file(destination_path):
    """Aligment of every point cloud in one Area"""
    print('[Corretcting Aligment of point clouds...]')

    angle_list = [f for f in os.listdir(destination_path) if f.endswith('.txt')]
    angle_file = angle_list[0]
    angle_path = os.path.join(destination_path, angle_file)
    room_list = [f for f in os.listdir(destination_path) if not f.endswith('.txt') and not f.endswith('re')]

    for room in room_list:
        print(f'[rotating the pc in the {room}...]')
        path = os.path.join(destination_path, room)
        rotate_one_pc(path, angle_path)

def rotate_one_pc(room_path, angle_path):
    """given a room path, it aligment the point cloud of this room with the angle in the aligment file"""
    # print(f'[rotating the point cloud of room {room_path}...]' )

    #finds the pc file
    room_list = [f for f in os.listdir(room_path) if f.endswith('.txt')]
    pc_file = room_list[0]
    room_name = pc_file.split('.')[0]
    room_name_rotated=room_name+ '_rotated.txt'

    point_cloud_path = os.path.join(room_path, pc_file)
    point_cloud_file = open(point_cloud_path, 'r')
    rotated_pc = os.path.join(room_path, room_name_rotated)
    rotated_file = open(rotated_pc, 'w')

    #get the translation vector
    point_cloud = pc.load_point_cloud(point_cloud_path)
    point_cloud_points = np.asarray(point_cloud.points)
    translation_vector = get_middle_point(point_cloud_points)

    #get the roation matrix
    angle = get_point_cloud_angle_aligment(angle_path, room_name)
    # rotation_matrix = get_rotation_matrix_around_Z(angle)
    # print('rotation matrix')
    # print(rotation_matrix)
    if angle == math.pi:
        rotation_matrix = np.matrix('-1 0 0; 0 -1 0 ; 0 0 1')
    if angle == 0:
        rotation_matrix = np.matrix('1 0 0; 0 1 0 ; 0 0 1')
    if angle == math.pi/2:
        rotation_matrix = np.matrix('0 -1 0; 1 0 0 ; 0 0 1')
    if angle == 3*math.pi/2:
        rotation_matrix = np.matrix('0 1 0; -1 0 0 ; 0 0 1')
    
    for line in point_cloud_file:
        ls = line.split(' ')
        x = ls[0:3]
        color = ls[3:6]
        pos = np.zeros(3)
        # print('--')
        pos[0] = float(x[0]) - translation_vector[0]
        pos[1] = float(x[1]) - translation_vector[1]
        pos[2] = float(x[2]) - translation_vector[2]
        # print(f'before rot: pos = {x[1]} - {translation_vector[1]}')
        x = rotation_matrix@pos
        pos[0] = float(x[0,0]) + translation_vector[0]
        pos[1] = float(x[0,1]) + translation_vector[1]
        # print(f'after rot: pos = {x[1]} + {translation_vector[1]}')
        pos[2] = float(x[0,2]) + translation_vector[2]
        setence = str(pos[0])+' '+str(pos[1])+' '+str(pos[2])+' '+str(color[0])+' '+ str(color[1]) + ' ' + str(color[2])+'\n'
        # print(setence)
        rotated_file.write(setence)
        # print(stop)
    
    point_cloud_file.close()
    rotated_file.close()

def get_middle_point(point_cloud_points):
    """return the middle of the point cloud"""
    X = point_cloud_points[:,0]
    Y = point_cloud_points[:,1]
    Z = point_cloud_points[:,2]
    # print(Y.max(), Y.min())
    X_med = (X.max()+X.min())/2
    Y_med = (Y.max()+Y.min())/2
    Z_med = (Z.max()+Z.min())/2

    P = np.array([X_med, Y_med, Z_med])

    # print(P)

    return P

def get_point_cloud_angle_aligment(angle_path, room_name):
    """given the path to the pc, returns the angle aligment for the room of that image in radians"""
    print('[getting the angle...]')
    f = open(angle_path)
    for line in f:
        if room_name in line:
            angle = int(line.split(' ')[-1])
            # print(angle)
    # angle = 270
    angle_rad = (angle*math.pi)/180.
    # print(angle_rad)
    return angle_rad

def organize_database(database_path, destination_path):
    create_room_folders(database_path, destination_path)
    populate_folders_with_point_cloud(database_path, destination_path)
    create_2dfolders(destination_path)
    popuate_folders_with_its_pose_files(database_path, destination_path)
    popuate_folders_with_its_color_images(database_path, destination_path)
    popuate_folders_with_its_depth_images(database_path, destination_path)
    
def populate_npy(path, area_number, destination_path):
    name= 'Area_'+str(area_number)
    original_path = os.path.join(path, name, 'coords' )
    dir_list = os.listdir(original_path)
    for room in dir_list:
        room_npy = str(room)
        room_name = room_npy.split('.')[0] 
        room_npy = room_name+'_aligned.npy'
        # print(room_name)
        organize_data_name = 'database_organized_Area'+str(area_number)
        src_path = os.path.join(original_path, room)
        dst_path = os.path.join(destination_path, organize_data_name, room_name, room_npy)
        # print(src_path)
        print(dst_path)
        shutil.copy(src_path, dst_path) 
   

if __name__ == "__main__":
    # path = 'database\database_Area3'
    path = 'old_Stanford3dDataset_v1.2\old_Stanford3dDataset_v1.2'
    destination_path = 'database_organized'
    

    # create_room_folders(path, destination_path)
    # populate_folders_with_point_cloud(path, destination_path)
    # create_2dfolders(destination_path)
    # popuate_folders_with_its_pose_files(path, destination_path)
    # popuate_folders_with_its_color_images(path, destination_path)
    # popuate_folders_with_its_depth_images(path, destination_path)
    # organize_database(path, destination_path)
    # rotate_the_point_cloud_file(destination_path)

    # populate_npy('npy_aligned', 1, 'database_organized')
    populate_folders_with_not_alig_point_cloud(path, 3, destination_path)
 
    # populate_folders_with_point_cloud_alig_not_alig(path, 3, destination_path)
