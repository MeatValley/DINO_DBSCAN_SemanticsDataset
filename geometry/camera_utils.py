import torch
import torch.nn.functional as funct
import numpy as np
import math


def construct_K(fx, fy, cx, cy, dtype=torch.float, device=None):
    """Construct a [3,3] camera intrinsics from pinhole parameters"""

    return torch.tensor([[fx,  0, cx],
                         [0, fy, cy],
                         [0,  0,  1]], dtype=dtype, device=device)


def scale_intrinsics(K, x_scale, y_scale):
    """Scale intrinsics given x_scale and y_scale factors"""
    K[..., 0, 0] *= x_scale
    K[..., 1, 1] *= y_scale
    K[..., 0, 2] *= x_scale
    K[..., 1, 2] *= y_scale
    return K

def rotation_matrix_Euler_angles(alfa, beta, gama):
    """Given the 3 Euler angles alfa,beta,gama in XYZ we can compute the Rotation matrix of them"""
    R_x = np.array((
        (1,0,0,0),
        (0, math.cos(alfa), (-1)*(math.sin(alfa)), 0),
        (0, math.sin(alfa), math.cos(alfa),0),
        (0,0,0,1)
        ))
    # print(R_x)
    R_y = np.array((
        ( math.cos(beta), 0, math.sin(beta),0 ),
        (0,1,0,0),
        ( (-1)*(math.sin(beta)) , 0, math.cos(beta),0 ),
        (0,0,0,1)
        ))
    # print(R_y)
    R_z = np.array((
        ( math.cos(gama),  (-1)*(math.sin(gama)), 0,0),
        ( math.sin(gama),  math.sin(gama), 0,0),
        (0,0,1,0),
        (0,0,0,1)
        ))
    # print(R_z)
    
    R = (R_z@R_y)@R_z

    # print(R)

    return R
def crop_intrinsics(intrinsics, borders):
    """
    Crop camera intrinsics matrix

    Parameters
    ----------
    intrinsics : np.array [3,3]
        Original intrinsics matrix
    borders : tuple
        Borders used for cropping (left, top, right, bottom)
    Returns
    -------
    intrinsics : np.array [3,3]
        Cropped intrinsics matrix
    """
    intrinsics = np.copy(intrinsics)
    intrinsics[0, 2] -= borders[0] #cx
    intrinsics[1, 2] -= borders[1] #cy
    return intrinsics


if __name__ == "__main__":
    R = rotation_matrix_Euler_angles(1.717,-0.00051,2.58701)
    print(R)
    original = np.array([[1.588232159614563], [-0.003562330733984709], [2.7595205307006836],[1]]) 
    print(original)
    print(R@original)