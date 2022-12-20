import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from myutils.folder_manipulation import organize_database
import numpy as np
from myutils.folder_manipulation import organize_database
from dataset.semantics_dataset import SemanticsDataset
import open3d as o3d


#after creates_area_folder, put every 3d file of area in 3d fodler of that area and same for 2d
#in the end must be something like database\database_Area1\2d\data\rgb\camera_0a70cd8d4f2b48239aaa5db59719158a_office_12_frame_1_domain_rgb.png
#or like database\database_Area1\3d\hallway_3\hallway_3.txt


def generate_features_file(area_path, destination_path, organize_files = False):
    """
    area_path: is where the "database_Area1,2,3,4,5,6" is
    """
    if organize_files: organize_database(area_path, destination_path)


def generate_boolean_image_mask(npy_path, image_path):
    """
    input: 
    npy_path - path to the npy points coord of that room
    image_path - path to the image desired

    output:
    a boolean mask with 
    
    """


    #creating the dataset
    print('[generetaing boolean mask...]')
    path = image_path.split('\\')
    core = path[-1]
    room_list = core.split('_')
    room_name = room_list[2]+'_'+room_list[3]
    room_path = ''
    point_cloud_name = room_name +'_not_alig.txt' #+ '_rotated.txt'
    for part in path:
        room_path = os.path.join(room_path, part)
        if part == room_name: break
    print(room_path)
    dataset = SemanticsDataset(room_path, point_cloud_name)


    b = dataset.get_one_boolean_image_mask(image_path)

    
    return b

def get_and_save_boolean_mask(image_path):
    b = generate_boolean_image_mask('', image_path)
    b = np.array(b).astype(np.uint8)
    
    print(b)

    path = image_path.split('\\')
    core = path[-1]
    room_list = core.split('.')
    save_npy_name = room_list[0]+'.npy'

    save_path = os.path.join('tests',save_npy_name)
    np.save(save_path,b)

def get_boolean_mask_for_one_room(npy_room_path, images_room_path, destination_path):
    """
    input: path to 
    """

    print('[generetaing boolean mask...]')
    image_list = os.listdir(images_room_path)

    path = npy_room_path.split('\\')
    npy_name = path[-1]
    room_name = npy_name.split('.')
    room_name = room_name[0]
    point_cloud_name = room_name +'_not_alig.txt'

    room_path = ''
    path = images_room_path.split('\\')
    # room_path = room_path[0:-2]

    for part in path:
        
        if part == 'color' or part == 'rgb': break
        room_path = os.path.join(room_path, part)
    
    print(point_cloud_name)
    print(room_path)
    dataset = SemanticsDataset(room_path, point_cloud_name)
    dataset.get_all_boolean_angles_for_a_room(destination_path)

if __name__ == "__main__":
    path = 'database_organized\database_organized_Area1\conferenceRoom_1\color\camera_0d600f92f8d14e288ddc590c32584a5a_conferenceRoom_1_frame_0_domain_rgb.png'
    # get_and_save_boolean_mask(path)
    dst_path = 'bool_mask\Area1\conferenceRoom_1'
    npy_path = 'database\S3DIS_processed_non_aligned\Area_1\coords\conferenceRoom_1.npy'
    image_path = 'database_organized\database_organized_Area1\conferenceRoom_1\color'
    get_boolean_mask_for_one_room(npy_path, image_path, dst_path)

    # x = np.load('bool_mask\Area1\conferenceRoom_1\camera_042a479869b44a7c9159922f19a285ea_conferenceRoom_1_frame_11_domain_rgb.npy')
    # pos = np.load('database\S3DIS_processed_non_aligned\Area_1\coords\conferenceRoom_1.npy')
    # col = np.load('database\S3DIS_processed_non_aligned\Area_1/rgb\conferenceRoom_1.npy')
    # c = np.zeros((len(x), 3))
    # e = np.zeros((len(x), 3))
    # for i in range(len(x)):
    #     if x[i]==True:
    #         c[i]=pos[i]
    #         e[i]=col[i]/255
    #     else:
    #         c[i]=pos[0]
    

    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(c)
    # point_cloud.colors = o3d.utility.Vector3dVector(e)
        
    # print('[vizualiation of pc...]')
    # vis = o3d.visualization.VisualizerWithEditing()
    # vis.create_window(width = 1300, height = 700)
    # vis.add_geometry(point_cloud)
    # vis.run()  # user picks points
    # vis.destroy_window()



