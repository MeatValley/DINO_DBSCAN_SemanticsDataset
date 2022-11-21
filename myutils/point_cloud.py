from json import load
import numpy as np
import open3d as o3d
import os

from sklearn.cluster import KMeans as kmeans

from sklearn.cluster import DBSCAN as dbscan
# from sklearn.preprocessing import StandardScaler

MAX_NUM_POINT = 20000

FIX_COLORS = np.array([
    [0,1,0], 
    [0,0,1], 
    [1,1,0], 
    [1,0,1], 
    [1,1,0], 
    [0,1,1],
    [0.5,1,0],
    [0,0.5,1],
    [1,0,0.5],
    [1,1,0.5],
    [0.5,1,1],
    [0.2, 0.2, 0.2],
    [0.9, 0.9,0.3],
    [0.3,0.3,0.9],
    [0.5,0.1, 0.2]
    ])

#################################################################################### - saving pc
def load_point_cloud(path):
    """ load point cloud from a file.
    
        Args:
            point_cloud: o3d.geometry.PointCloud
        
        Returns:
            o3d.geometry.PointCloud
    """
    if path.endswith('.ply'):
        point_cloud_loaded = o3d.io.read_point_cloud(path)
        return point_cloud_loaded
    if path.endswith('.txt'):
        print('[reading a point cloud from a txt...]')
        point_cloud_loaded = o3d.io.read_point_cloud(path, format="xyzrgb")
        return point_cloud_loaded

def save_point_cloud(point_cloud, path):
    """ Save a point cloud to the specified path.
    
    Args:
        point_cloud: o3d.geometry.PointCloud
        path: str
        
    Returns:
        None
    """
    o3d.io.write_point_cloud(path, point_cloud)

def save_point_cloud_with_labels(points, labels, config, K, n_img):
    
    point_colors = [FIX_COLORS[labels[i]] for i in range(len(labels))] #for each point in pc (len labels)
   
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(point_colors)

    path = 'temp_pc'
    final_path = os.path.join(path, config.experiment.name+ '_' + str(K) + 'mean'+'_' + str(n_img) +'imagesUsed' +'_' + str(config.experiment.time)+'_point_cloud.ply')
    print(final_path)
    o3d.io.write_point_cloud(final_path, point_cloud)
    return

#################################################################################### - show pc
def show_point_clouds(point_clouds):
    """ Show a list of point clouds.
    
        Args:
            point_clouds: list of o3d.geometry.PointCloud
        
        Returns:
            None
    """
    o3d.visualization.draw_geometries(point_clouds, width = 1500, height = 800)

def show_point_cloud(point_cloud,  window_name="Point Cloud vizualization", width = 1500, height = 800):
    """ Show a list of point clouds.
    
        Args:
            point_clouds: o3d.geometry.PointCloud
        
        Returns:
            None
    """

    point_clouds = [point_cloud]
    window_name = window_name
        
    o3d.visualization.draw_geometries(point_clouds, window_name, width , height)

def show_point_clouds_with_labels(point_clouds_np, labels, random_colors = False):

    print('[showing point cloud with labels...]')
    """ Show a list of point clouds with their labels.
    
        Args:
            point_clouds_np: list of numpy arrays of shape (n, d)
            labels: list of numpy arrays of shape (n,), same of points in the same cluster
        
        Returns:
            o3d.geometry.PointCloud
    """

    colors = np.random.rand(len(labels), 3) #for each pixel
    # [0.15 0.85 0.32] normalzied colors in rgb
    
    if random_colors:
        point_colors = [colors[labels[i]] for i in range(len(labels))] #for each point in pc (len labels)
    else:
        point_colors = [FIX_COLORS[labels[i]] for i in range(len(labels))] #for each point in pc (len labels)
   
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(point_clouds_np)
    point_cloud.colors = o3d.utility.Vector3dVector(point_colors)
    
    show_point_cloud(point_cloud)
    return point_cloud

def show_point_clouds_separeted_by_color(point_cloud, save_pc = False):
    monocolor_point_clouds = get_point_clouds_separeted_by_color(point_cloud)
    color = 'color'
    i = 0
    for pc in monocolor_point_clouds:
        i+=1
        path = 'temp_pc/pc_separeted_by_color'+'_'+color+str(i)+'.ply'
        show_point_cloud(pc)
        if save_pc: save_point_cloud(pc, path)

#################################################################################### - filtering pc
def get_point_cloud_by_color(point_cloud, color):
    """receive a pc and return a point cloud with the points with that color
    
    input: o3d.geometry.PointCloud, color
    output: o3d.geometry.PointCloud with that color
    
    """
    print(f'[filtering for color {color} ...]')
    sub_point_cloud_points = o3d.utility.Vector3dVector()
    sub_point_cloud_color = o3d.utility.Vector3dVector()
    

    for key, point in enumerate(point_cloud.points):
        if (point_cloud.colors[key] == color).all(): 
            sub_point_cloud_color.append(color)
            sub_point_cloud_points.append(point)


    sub_point_cloud = o3d.geometry.PointCloud()
    sub_point_cloud.points = sub_point_cloud_points
    sub_point_cloud.colors = sub_point_cloud_color
    return sub_point_cloud

def get_point_cloud_colors(point_cloud):
    """return a vector with the rgb of the colors presents in this point cloud"""
    colors = []
    print('[getting pc colors...]')
   
    # show_point_cloud(point_cloud)
    res_list = []
    mylist = list(point_cloud.colors)

    for item in mylist:
        if item not in res_list:
            res_list.append(item)

    print(res_list)


    # points = np.array(point_cloud.points)
    # # print(points.shape)
    # put = True
    # x = 0
    # print('[getting pc colors...]')
    # for key, point in enumerate(point_cloud.points):
    #     # print('key ', key)

    #     if key > 208361-100:
    #         break

    #     if key == 0: 
    #         colors.append(point_cloud.colors[key])
    #         # print('key 0')

    #     else:
    #         # print('else')
    #         for color in colors:
    #             # print('colors by now', colors)
    #             if (color == point_cloud.colors[key]).all():
    #                 # print('setting put as false')
    #                 put = False

    #         if put: 
    #             # print('putting a new color ', point_cloud.colors[key])
    #             colors.append(point_cloud.colors[key])

    #         put=True
    
    # print('returning colors')
    colors = res_list
    print(f'returning colors {len(colors)} ')
    return colors

def get_point_cloud_colors2(point_cloud):
    colors = np.array([
    [0,1,0], 
    [0,0,1], 
    [1,1,0], 
    [1,0,1], 
    [1,1,0]])
    return colors
def get_point_clouds_separeted_by_color(point_cloud):
    """receive a point cloud and return all the clusterings in a list of o3d.geometry.PointCloud """
    print('[getting pc separated by color...]')
    colors = get_point_cloud_colors(point_cloud)
    monocolor_point_clouds = []
    print('[running color in colors on get separated by color...]')
    for color in colors:
        pc = get_point_cloud_by_color(point_cloud, color)
        monocolor_point_clouds.append(pc)
    
    return monocolor_point_clouds

def reduce_point_cloud(point_cloud, max_points = 0):

    points = np.array(point_cloud.points)
    colors = np.array(point_cloud.colors)
    
    
    N = points.shape[0]
    if N<10000: return point_cloud #camon, 10000 is okay

    if max_points == 0: max_points = int(N*0.2)
    if N > max_points:
        choices = np.random.choice(N, max_points, replace=False)
        new_points = points[choices, :]
        sub_point_cloud_points = o3d.utility.Vector3dVector(new_points)
        new_colors = colors[choices, :]
        sub_point_cloud_color = o3d.utility.Vector3dVector(new_colors)
        # print(new_colors)

        sub_point_cloud = o3d.geometry.PointCloud()
        sub_point_cloud.points = sub_point_cloud_points
        sub_point_cloud.colors = sub_point_cloud_color
        # show_point_cloud(sub_point_cloud)

        return sub_point_cloud

# def test_with_DBSCAN_by_pc(point_cloud):
#     print('[running DBSCAN receiving a point cloud...]')

#     points = point_cloud.points
#     points = np.array(points)
#     clustering = DBSCAN(eps=10, min_samples=30).fit(points)
#     labels = clustering.labels_
#     n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#     print('number of clusters ', n_clusters_)
#     ret_pc = pc.show_point_clouds_with_labels(points, list(labels), random_colors=True)
#     return ret_pc

def testing_dbscan(point_cloud):
    print('[testing bounding box with DBSCAN...]')
    # pck6 = reduce_point_cloud(pck6)
    colors = get_point_cloud_colors(point_cloud)
    print(colors)
    for color in colors:
        pc = get_point_cloud_by_color(point_cloud, color)
        show_point_cloud(pc)
        points = np.array(pc.points)
        print(points)
        # clustering = DBSCAN(eps=10, min_samples=30).fit(points)
        # pc_dbscaned = test_with_DBSCAN_by_pc(point_cloud)


if __name__ == "__main__":
    path = "data/3d/Area_1/conferenceRoom_1/conferenceRoom_1.txt"
    
    # convert_xyzrgb_to_ply(path)

    # path2 = 'trash/teste1.txt'
    # pcd = o3d.io.read_point_cloud(path, format="xyz")
    # o3d.io.write_point_cloud("output.ply", pcd)

    pc = load_point_cloud('temp_pc/1image_default.ply')
    pck6 = load_point_cloud('temp_pc\conferenceRoom1_6mean_1imagesUsed_2022-11-16-09-51_point_cloud.ply')

    # pck6 = reduce_point_cloud(pck6)
    # colors = get_point_cloud_colors(pck6)
    # print(colors)
    # for color in colors:
    #     pc = get_point_cloud_by_color(pck6, color)
    #     show_point_cloud(pc)
    testing_dbscan(pck6)