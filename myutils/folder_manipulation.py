import os
import shutil


def create_folders(database_path, destination_path):
    """
    receive the path to the database folder->
    
    returns a Area1 folder with each room with each point cloud
    
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
    """populate each fodler with its pointcloud"""
    original_path = os.path.join(database_path, '3d')
    dir_list = os.listdir(original_path)
    for room in dir_list:
        if room != '.DS_Store': 
            room_name = str(room)
            point_cloud_name = room_name+'.txt'
            src_path = os.path.join(original_path, room, point_cloud_name )
            dst_path = os.path.join(destination_path, room)
            shutil.copy(src_path, dst_path) 
   
def create_2dfolders(destination_path):
    """
    receive the path to the database folder->
    
    creates a rgb, pose and depth folder for each room
    
    """
    dir_list = os.listdir(destination_path)
    try:
        for room in dir_list:
            print(room)
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
    
    creates a rgb, pose and depth folder for each room
    
    """
    original_path = os.path.join(database_path, '2d')
    original_path = os.path.join(original_path, 'data', 'pose')
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
    original_path = os.path.join(original_path, 'data', 'rgb')
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
    original_path = os.path.join(original_path, 'data', 'depth')
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
          
def organize_database(database_path, destination_path):
    create_folders(database_path, destination_path)
    populate_folders_with_point_cloud(database_path, destination_path)
    create_2dfolders(destination_path)
    popuate_folders_with_its_pose_files(database_path, destination_path)
    popuate_folders_with_its_color_images(database_path, destination_path)
    popuate_folders_with_its_depth_images(database_path, destination_path)

if __name__ == "__main__":
    path = 'database\database_Area1'
    destination_path = 'database_organized\database_organized_Area1'
    # create_folders(path, destination_path)
    # populate_folders_with_point_cloud(path, destination_path)
    # create_2dfolders(destinantion_path)
    # popuate_folders_with_its_pose_files(path, destination_path)
    # popuate_folders_with_its_color_images(path, destination_path)
    # popuate_folders_with_its_depth_images(path, destination_path)
    organize_database(path, destination_path)