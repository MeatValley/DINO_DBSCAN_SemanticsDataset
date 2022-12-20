import matplotlib.pyplot as plt
import feature_extractor.DINO as DINO
from geometry.camera import Camera
from myutils.image import load_image, plot
import myutils.point_cloud as pc
import numpy as np
import os
import open3d as o3d
import math

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from geometry.camera_utils import get_rotation_matrix_around_X, get_rotation_matrix_around_Z, get_rotation_matrix_around_Y

########################################################################################################################
# 2d-3d semantics dataset
########################################################################################################################
class Camara_frame():
    print('[creating a camera frame...]')
    def __init__(self, uid, K, frame, rt_matrix, room, angles, RT, loc):
        self.uid = uid
        self.K = K
        self.frame = frame
        self.rotation_matrix = rt_matrix
        self.room = room
        self.angles = angles
        self.RT = RT
        self.loc = loc
        # print(f'uid {self.uid}\n, k {self.K}, frame{self.frame}')


class SemanticsDataset(Dataset):
    print('[creating a dataset...]')

    def __init__(self, root_dir, point_cloud_name, stop_image=1, add_patch=None, features="dino", device="cpu"):

        self.features = features
        self.add_patch = add_patch
        self.stop = stop_image
        self.scale = 336
        if self.stop == 0:
            print('[you are not reading any image!]')

        point_cloud_path = os.path.join(root_dir, point_cloud_name)
        npy_name = point_cloud_name.split('_')
        aux = npy_name[1]
        aux = aux.split('.')[0]
        # print(aux)
        npy_name = npy_name[0]+'_'+ aux+'_not_aligned.npy'
        
        # print(npy_name)
        # npy_path = os.path.join(root_dir, npy_name)
        npy_path = 'database\S3DIS_processed_non_aligned\Area_1\coords\conferenceRoom_1.npy' ####################
        # print(npy_path)

        self.data_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

 
        # get all images in the folder stream from "color" and with "depth"

        # list of color images

        # join the str in a path
        image_dir = os.path.join(root_dir, 'color')
        self.color_split = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                            if os.path.isfile(os.path.join(image_dir, f)) and
                            (f.endswith('.png') or f.endswith('.jpg')
                                or f.endswith('.jpeg'))]

        # #list of detph images
        depth_dir = os.path.join(root_dir, 'depth')
        self.depth_split = [os.path.join(
            depth_dir, f) for f in os.listdir(depth_dir)]
        # print(self.depth_split)

        # #list of poses matrix
        pose_dir = os.path.join(root_dir, 'pose')
        self.pose_var_split = [os.path.join(pose_dir, f) for f in os.listdir(pose_dir)
                                if os.path.isfile(os.path.join(pose_dir, f)) and
                                f.endswith('.json')]
        self.pose_dir = pose_dir
        # print(self.pose_split)

        self.camera_k_matrix_split = []
        self.pose_split = []
        self.camera_frame = []
        for json_file in self.pose_var_split:
            cam_f = self.get_pose(json_file)
            self.camera_k_matrix_split.append(cam_f.K)
            self.pose_split.append(cam_f.rotation_matrix)
            self.camera_frame.append(cam_f)

        self.root_dir = root_dir

        # #K for color img
        self.intrinsic = self.camera_k_matrix_split

        # #K for depth img
        self.intrinsic_depth = self.camera_k_matrix_split

        # point_cloud in the data
        self.point_cloud = pc.load_point_cloud(point_cloud_path)
        print('retrinving pc in: ', point_cloud_path)
        self.point_cloud_points = np.asarray(self.point_cloud.points)
        self.point_cloud_colors = np.asarray(self.point_cloud.colors)
        self.point_cloud_npy_ordinate = np.load(npy_path)


        self.device = torch.device(device)
        self.dino_model, self.patch_size = DINO.get_model('vits8')
        self.dino_model = self.dino_model.to(self.device)

        self.preprocess = transforms.Compose([
            transforms.Resize(
                self.scale, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.FiveCrop(224),
        ])
        self.normalize_preprocess = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
                                                #((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


    def get_pose(self, json_path):
        """given a path to a json file, return a camera_frame with all the informations"""

        f = open(json_path)
        # print(json_path)
        # f = open('database_organized\database_organized_Area1\conferenceRoom_2\pose\camera_a223e73ba60140b8ac0c72841a70a772_conferenceRoom_2_frame_5_domain_pose.json')
        data = json.load(f)
        # print(data)

        aux = data['camera_k_matrix']
        uid = data['point_uuid']
        frame = data['frame_num']
        room = data["room"]
        angles = data["final_camera_rotation"]
        loc = data["camera_location"]
        
        aux.append([0., 0., 0.])
        aux = np.array(aux)
        aux = np.concatenate((aux, [[0.], [0.], [0.], [1.]]), axis=1)

        temp = data['camera_rt_matrix']
        
        
        alfa = data['camera_rt_matrix']
        alfa = np.array(alfa)
        # print(alfa.shape)
        # print(temp)
        temp.append([0, 0, 0, 1])
        # print(temp)
        
        cam_f = Camara_frame(uid, aux, frame, temp, room, angles, alfa,loc)
        return cam_f
    
    def get_depth_path(self, image_path):
        path = image_path.split('\\')
        core = path[-1]
        room_list = core.split('_')
        room_name = room_list[2]+'_'+room_list[3]
        depth_path = ''
        for part in path:
            depth_path = os.path.join(depth_path, part)
            if part == room_name: break
        
        core_list = core.split('_')
        core = core_list[0]+'_'+core_list[1]+'_'+core_list[2]+'_'+core_list[3]+'_'+core_list[4]+'_'+core_list[5]+'_'+core_list[6]+'_'+'depth.png'
        depth_path = os.path.join(depth_path, 'depth', core)


        return depth_path

    def get_pose_path(self, image_path):
        path = image_path.split('\\')
        core = path[-1]
        room_list = core.split('_')
        room_name = room_list[2]+'_'+room_list[3]
        pose_path = ''
        for part in path:
            pose_path = os.path.join(pose_path, part)
            if part == room_name: break
        
        core_list = core.split('_')
        core = core_list[0]+'_'+core_list[1]+'_'+core_list[2]+'_'+core_list[3]+'_'+core_list[4]+'_'+core_list[5]+'_'+core_list[6]+'_'+'pose.json'
        pose_path = os.path.join(pose_path, 'pose', core)

        return pose_path

    def get_middle_point(self, point_cloud_points):
        """return the middle of the point cloud"""
        X = point_cloud_points[:,0]
        Y = point_cloud_points[:,1]
        Z = point_cloud_points[:,2]
        # print(Y.max(), Y.min())
        X_med = (X.max()+X.min())/2
        Y_med = (Y.max()+Y.min())/2
        Z_med = (Z.max()+Z.min())/2

        P = np.array([X_med, Y_med, Z_med])

        print(P)

        return P

    def get_an_image_point_cloud(self, image_path):
        print('[retrieving the point cloud for a given image...]')
        depth_path = self.get_depth_path(image_path)
        pose_path = self.get_pose_path(image_path)
    
        cam_f = self.get_pose(pose_path)

        image = load_image(image_path)
        depth = load_image(depth_path)
        dims_image = tuple(np.asarray(image).shape[0:2][::-1])

        depth = np.array(depth)/512  # in meter
        depth[depth > 65534/512] = 0.0001
        plot(image, image_path)
        # plot(depth, 'original depth image')
        
        pose  = cam_f.rotation_matrix
        camera_intrinsic = cam_f.K

        cam = Camera(K=camera_intrinsic, dimensions=dims_image, Tcw=np.linalg.inv(pose))
 
        coord_depth = np.asarray([[i, j] for i in range(depth.shape[0]) for j in range(depth.shape[1])])  # 1080*1080*2 xy for 1080^2
        value_depth = np.asarray([[depth[i, j]] for i in range(depth.shape[0]) for j in range(depth.shape[1])])
        point_cloud_np = cam.get_point_cloud(coord_depth, value_depth)  # (x,y,z, 1080*1080) point_xyz_world

        # point_cloudx = o3d.geometry.PointCloud()
        # point_cloudx.points = o3d.utility.Vector3dVector(point_cloud_np)
        # o3d.visualization.draw_geometries([point_cloudx], width = 1500, height = 800)

        projection3dto2d_depth = cam.project_on_image(X_pos=point_cloud_np, X_color=value_depth)
        # plot(projection3dto2d_depth, 'projection calculated') #good

        p = self.point_cloud_npy_ordinate
        projection3dto2d, pc_im_correspondances = cam.project_on_image(X_pos=p, X_color=self.point_cloud_colors, depth_map=projection3dto2d_depth, corres=True, eps=0.05)
        plot(projection3dto2d, 'image calculated')

        points = []
        colors = []

        for j, point in enumerate(self.point_cloud_points):
            if j in pc_im_correspondances.keys():
  
                point = np.asarray(point)
                color = self.point_cloud_colors[j]/255
                points.append(point[None, :]) 
                colors.append(color[None, :])
        
        points = np.concatenate(points, axis=0) #axis = 0 treats each independent
        colors = np.concatenate(colors, axis=0)

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        pc.show_point_cloud(point_cloud)

        return point_cloud

    def get_one_boolean_image_mask(self, image_path):
        print('[retrieving the point cloud for a given image...]')
        depth_path = self.get_depth_path(image_path)
        pose_path = self.get_pose_path(image_path)
    
        cam_f = self.get_pose(pose_path)

        image = load_image(image_path)
        depth = load_image(depth_path)
        dims_image = tuple(np.asarray(image).shape[0:2][::-1])

        depth = np.array(depth)/512  # in meter
        depth[depth > 65534/512] = 0.0001
        # plot(image, image_path)
        # plot(depth, 'original depth image')
        
        pose  = cam_f.rotation_matrix
        camera_intrinsic = cam_f.K

        cam = Camera(K=camera_intrinsic, dimensions=dims_image, Tcw=np.linalg.inv(pose))
 
        coord_depth = np.asarray([[i, j] for i in range(depth.shape[0]) for j in range(depth.shape[1])])  # 1080*1080*2 xy for 1080^2
        value_depth = np.asarray([[depth[i, j]] for i in range(depth.shape[0]) for j in range(depth.shape[1])])
        point_cloud_np = cam.get_point_cloud(coord_depth, value_depth)  # (x,y,z, 1080*1080) point_xyz_world

        # point_cloudx = o3d.geometry.PointCloud()
        # point_cloudx.points = o3d.utility.Vector3dVector(point_cloud_np)
        # o3d.visualization.draw_geometries([point_cloudx], width = 1500, height = 800)

        projection3dto2d_depth = cam.project_on_image(X_pos=point_cloud_np, X_color=value_depth)
        # plot(projection3dto2d_depth, 'projection calculated') #good

        
        points = self.point_cloud_points
        points_npy = self.point_cloud_npy_ordinate
        # print(points)
        # projection3dto2d, pc_im_correspondances = cam.project_on_image(X_pos=points_npy, X_color=self.point_cloud_colors, depth_map=projection3dto2d_depth, corres=True, eps=0.05)
        # plot(projection3dto2d, 'image calculated')
        b, pc_im_correspondances = cam.get_boolean_vector(points=points_npy, depth_map=projection3dto2d_depth )
        
        # print(b)
        # points = []
        # colors = []

        # for j, point in enumerate(self.point_cloud_points):
        # for j, point in enumerate(self.point_cloud_npy_ordinate):
        #     if j in pc_im_correspondances.keys():
  
        #         point = np.asarray(point)
        #         color = self.point_cloud_colors[j]/255
        #         points.append(point[None, :]) 
        #         colors.append(color[None, :])
        
        # points = np.concatenate(points, axis=0) #axis = 0 treats each independent
        # colors = np.concatenate(colors, axis=0)

        # point_cloud = o3d.geometry.PointCloud()
        # point_cloud.points = o3d.utility.Vector3dVector(points)
        # point_cloud.colors = o3d.utility.Vector3dVector(colors)
        # pc.show_point_cloud(point_cloud)

        return b

    def get_room_name(self, image_path):
        path = image_path.split('\\')
        core = path[-1]
        room_list = core.split('_')
        room_name = room_list[2]+'_'+room_list[3]
        return room_name

    def get_all_boolean_angles_for_a_room(self, save_path):
        for img in self.color_split:
            # print(img)
            core = img.split('\\')
            core = core[-1]
            core = core.split('.')
            core = core[0]
            print(core)
            b = self.get_one_boolean_image_mask(img)
            b = np.array(b).astype(np.uint8)
            dst = os.path.join(save_path, core)
            np.save(dst,b)
            

    def get_point_cloud_angle_aligment(self, image_path):
        """given the path to te image, returns the angle aligment for the room of that image in radians"""
        print('[getting the angle...]')
        room_name = self.get_room_name(image_path)
        # print('room ', room_name)
        f = open('database_organized\database_organized_Area1\Area_1_alignmentAngle.txt')
        for line in f:
            if room_name in line:
                angle = int(line.split(' ')[-1])
                # print(angle)
        # angle = 270
        angle_rad = (angle*math.pi)/180.
        # print(angle_rad)
        return angle_rad

    def adjust_point_cloud(self, image_path):
        print('[adjusting the pc...]')
        # print(self.point_cloud_points)
        angle = self.get_point_cloud_angle_aligment(image_path)
        # print(angle)
        Rz = get_rotation_matrix_around_Z(angle) 
        self.point_cloud_points = (Rz@(self.point_cloud_points).T).T
        # print(self.point_cloud_points)
    
    def adjust_cam(self, cam_f, P, angle):
        print('[adjusting cam...]')
        """given an angle, adjust the camera aligment"""
        Rz = get_rotation_matrix_around_Z(angle) #
        RT = cam_f.rotation_matrix #
        R = np.array(RT)[:-1, :-1] #

        new_loc = cam_f.loc - P #world frame
        new_loc_cam = -(R@new_loc) #camera frame
        new_RT = self.translate_RT(RT, new_loc_cam) #camera translation
        new_RT = new_RT[:-1, :] #

        new_RT = Rz@new_RT #

        pos  = new_RT[:, -1] #
        pos = np.linalg.inv(R)@pos
        pos = pos+P
        pos = R@pos

        new_RT = self.translate_RT(new_RT, pos)

        # print('new rt \n', new_RT)
        return new_RT

    def rotate_point_cloud(self, path):
        f = open(path, "r")

    def translate_RT(self, RT, T):
        print('[translading...]')
        """adds T to las column of RT"""
        new_RT = np.array(RT)
        new_RT[0, -1] = T[0]
        new_RT[1, -1] = T[1]
        new_RT[2, -1] = T[2]
        print(new_RT)
        return new_RT

########################################################################################################################

    @staticmethod
    def _get_pose(pose_file):
        return np.loadtxt(pose_file)

# ########################################################################################################################
    def __len__(self):
        """Dataset length"""
        return len(self.color_split)

########################################################################################################################
    @staticmethod
    def _get_intrinsics(intrinsic_file):
        """Get intrinsics from the calib_data dictionary."""
        return np.loadtxt(intrinsic_file)


########################################################################################################################

    def __getitem__(self, index_):
        """Get dataset sample given an index."""
        index = index_+3
        print(f'[getting a sample for the dataset with {index} index... ]')

        image = load_image(self.color_split[index])  # 1080x1080

        depth = load_image(self.depth_split[index])
        plot(image, self.color_split[index])

        image_name = (self.color_split[index]).split('_')
        camera_name = 'camera' + '_' + image_name[5] + '_' + image_name[6]  + '_' + image_name[7] 
        camera_name += '_' + image_name[8] + '_' + image_name[9] + '_domain_pose.json'
        
        pose_json_path = os.path.join(self.pose_dir, camera_name)
        # print(pose_json_path)
        cam_f = self.get_pose(pose_json_path)

        # dimensions
        dims_image = tuple(np.asarray(image).shape[0:2][::-1])
        dims_depth = tuple(np.asarray(depth).shape[0:2][::-1])

        image = image.crop((20, 20, 1080-20, 1080-20))  # is needed

        original_image = image  # 1040x1040
        image_5crops = self.preprocess(image) #img


        image_res = image.resize((self.scale, self.scale))  # 1080*0.4 = 432
        width, height = image_res.size

        depth = np.array(depth)/512  # in meter
        depth[depth > 65534/512] = -0.1
        # plot(depth, self.depth_split[index])

        # get the pose of the current image
        # pose = np.array(self.pose_split[index])
        pose  = cam_f.rotation_matrix
        # print(self.pose_split[index])
        # camera_intrinsic = np.array(self.camera_k_matrix_split[index])
        camera_intrinsic = cam_f.K

        H, W = 224, 224  # size for dino
        new_width = W
        new_height = H

        crops = [(0, 0, new_width, new_height),
                 (width - new_width, 0, width, new_height),
                 (0, height - new_height, new_width, height),
                 (width - new_width, height - new_height, width, height),
                 ((width - new_width)/2, (height - new_height)/2,
                  (width + new_width)/2, (height + new_height)/2)
                 ]

        image_crop_tot = []
        projection3dto2d_tot = []
        pose_tot = []
        pc_im_correspondances_tot = []
        image_DINO_features_tot = []
        depth_tot = []
        projection3dto2d_depth_tot = []
        features_interpolation_tot = []
        
        for n, image_ts in enumerate(list(image_5crops)):
            print(f'[doing the process to the {n} crop...]')
            # plot(image_ts, 'image_ts')
            image_ts = self.normalize_preprocess(image_ts)

            (left, top, right, bottom) = crops[n]
            image_crop = image_res.crop((left, top, right, bottom)) #image with 224, 224

            image_DINO_features_ts = DINO.get_DINO_features(self.dino_model.to('cpu'), image_ts)
            image_DINO_features = image_DINO_features_ts.detach().cpu().numpy().reshape(W//self.patch_size, H//self.patch_size, -1)

            # print('creating camera')
            cam = Camera(K=camera_intrinsic, dimensions=dims_image, Tcw=np.linalg.inv(pose)).scaled(self.scale/1080.).crop((left, top, right, bottom))
            cam_depth = Camera(K=camera_intrinsic, dimensions=dims_depth, Tcw=np.linalg.inv(pose))

            coord_depth = np.asarray([[i, j] for i in range(depth.shape[0]) for j in range(depth.shape[1])])  # 1080*1080*2 xy for 1080^2
            value_depth = np.asarray([[depth[i, j]] for i in range(depth.shape[0]) for j in range(depth.shape[1])])
            point_cloud_np = cam_depth.get_point_cloud(coord_depth, value_depth)  # (x,y,z, 1080*1080) point_xyz_world

            # point_cloudx = o3d.geometry.PointCloud()
            # point_cloudx.points = o3d.utility.Vector3dVector(point_cloud_np)
            # o3d.visualization.draw_geometries([point_cloudx], width = 1500, height = 800)
            # (224,224,1) for each pixel of the img that goes dino, its depth

            projection3dto2d_depth = cam.project_on_image(X_pos=point_cloud_np, X_color=value_depth)
            # 224x224x1

            projection3dto2d, pc_im_correspondances = cam.project_on_image(X_pos=self.point_cloud_points, X_color=self.point_cloud_colors, depth_map=projection3dto2d_depth, corres=True, eps=0.05)

            features_interpolation = None
            pose_tot.append(pose)
            image_crop_tot.append(np.asarray(image_crop)/255.0)
            depth_tot.append(depth)
            image_DINO_features_tot.append(image_DINO_features)
            features_interpolation_tot.append(features_interpolation)
            projection3dto2d_depth_tot.append(projection3dto2d_depth)
            projection3dto2d_tot.append(projection3dto2d)
            pc_im_correspondances_tot.append(pc_im_correspondances)

        print('[returning sample...]')
        sample = {
            "original_image": original_image,
            'pose': pose_tot,  # all TCW
            'image': image_crop_tot,  # all img with 224x224
            # all (28,28,384) tuples of patch & feature
            "image_DINO_features": image_DINO_features_tot,
            "depth_map": depth_tot,  # each z coordinate
            'proj3dto2d_depth': projection3dto2d_depth_tot,
            "feature_interpolation": features_interpolation_tot,
            'proj3dto2d': projection3dto2d_tot,
            'correspondances': pc_im_correspondances_tot,
            'path':self.color_split[index]

        }

        return sample