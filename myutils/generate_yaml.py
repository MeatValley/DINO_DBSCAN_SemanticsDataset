from pathlib import Path
import sys
import os
import yaml

def generate_room_yaml(room_path):
    room_name = room_path.split('\\')[-1]
    area_name = (room_path.split('\\')[-1]).split('_')[-1]


    print(room_name)
    BASE_PATH = Path().resolve() 
    print(BASE_PATH)
    args = {
        'experiment':{'name':room_name, 'area':'Area'+area_name},
        'pipeline':{'clustering':{'algo':'kmeans', 'k':'2', 'init_centroids':'++'}, 'feature_extractor':{'network':'DINO', 'model':'vits8'}},
        'data':{'name':'["Stanford3dDataset"]', 'path':room_path, 'point_cloud_name':room_name+'.txt', 'point_cloud':{'path':room_path+'/'+room_name+'.txt'}},
        'save':{'folder':'configs/logs', 'point_cloud': True, 'images': True}
    }

    (BASE_PATH / 'configs' / f'{room_name}.yaml').write_text(yaml.dump(args, default_flow_style=False, sort_keys=False))
    room_name+='.yaml'
    yaml_path = os.path.join('configs', room_name)
    # print(yaml_path)
    return yaml_path

# generate_room_yaml('database_organized\database_organized_Area1\conferenceRoom_1')