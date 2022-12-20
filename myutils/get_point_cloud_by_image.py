import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dataset.semantics_dataset import SemanticsDataset



def get_point_cloud_by_image(image_path_):

    image_path = image_path_.replace("/", "\\")
    path = image_path.split('\\')
    # path = image_path.split('/')
    core = path[-1]
    room_list = core.split('_')
    room_name = room_list[2]+'_'+room_list[3]
    room_path = ''
    point_cloud_name = room_name +'_not_alig.txt' #just txt is alig version in a1
    print('reading: ', point_cloud_name)
    for part in path:
        room_path = os.path.join(room_path, part)
        if part == room_name: break

    # print(room_path, point_cloud_name)image.png
    dataset = SemanticsDataset(room_path, point_cloud_name)
    dataset.get_an_image_point_cloud(image_path)
    




get_point_cloud_by_image('database_organized\database_organized_Area1\conferenceRoom_1\color\camera_0d600f92f8d14e288ddc590c32584a5a_conferenceRoom_1_frame_0_domain_rgb.png')
# get_point_cloud_by_image('database_organized\database_organized_Area3\lounge_1\color\camera_4a7bfe0577f74a1a891683cf5b435f93_lounge_1_frame_10_domain_rgb.png')
# get_point_cloud_by_image('database_organized\database_organized_Area2\auditorium_1\color\camera_fce7ddcfc0b64999b265e92d18950a64_auditorium_1_frame_30_domain_rgb.png')

###################### disjoint problem
# get_point_cloud_by_image('database_organized\database_organized_Area3\hallway_6\color\camera_448c589ba4a54cee90a91642303bb733_hallway_6_frame_10_domain_rgb.png')
# get_point_cloud_by_image('database_organized\database_organized_Area3\lounge_1\color\camera_4a7bfe0577f74a1a891683cf5b435f93_lounge_1_frame_10_domain_rgb.png')