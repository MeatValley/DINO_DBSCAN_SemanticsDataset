import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import myutils.point_cloud as pc
from myutils.parse import parse_args, get_default_config, merge_cfg_file
from myutils.folder_manipulation import organize_database
from myutils.generate_yaml import generate_yaml
import numpy as np

from run import run

if __name__ == "__main__":

    print('[main starting ...]')
    # args = parse_args()
    # organize_database('database\database_Area1', 'database_organized\database_organized_Area1')
    # cfg_file = generate_yaml('database_organized\database_organized_Area1')
    cfg_file = 'configs\conference_room01.yaml'
    
    run(cfg_file, number_images = 4, save_pc=False, K=9, max_pooling = True, run_complete = True)
   