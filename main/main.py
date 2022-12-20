import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import myutils.point_cloud as pc
from myutils.parse import parse_args, get_default_config, merge_cfg_file
from myutils.folder_manipulation import organize_database
from myutils.generate_yaml import generate_room_yaml
from run_spectral_clustering import run_spectral_clustering
import numpy as np

from run import run
def run_all_night():
    cfg_file = 'configs\conference_room01.yaml'
    
    run(cfg_file, number_images = 1, save_pc=True, K=16, robust_mean = True , max_pooling = False, run_complete = True)
    run(cfg_file, number_images = 1, save_pc=True, K=16, robust_mean = False , max_pooling = True, run_complete = True)
    

if __name__ == "__main__":
    print('[main starting ...]')
    # organize_database('database\database_Area1', 'database_organized\database_organized_Area1')
    
    cfg_file = generate_room_yaml('database_organized\database_organized_Area1\conferenceRoom_2')
    run(cfg_file, number_images = 2, save_pc=True, K=6, robust_mean = True , max_pooling = False, run_complete = False, show2d=False)
    # run_spectral_clustering(cfg_file, number_images = 2, save_pc=True, K=8, robust_mean = True , max_pooling = False, run_complete = False, show2d=False)


"""
images with bb
1) run with the image and save the pc
2) find path () to this image
3) obtain a PC with KMENAS

macaco
1 - run with path image
2 - copy path
3 - image with bb get_bounding_boxed_image(path, pck6)
"""