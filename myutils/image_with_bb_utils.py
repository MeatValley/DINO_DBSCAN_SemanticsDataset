import math
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from myutils.image import load_image, plot

def middle(p1, p2):
    p3 = p1
    p3[0] = (p1[0]+p2[0])/2
    p3[1] = (p1[1]+p2[1])/2
    p3[2] = (p1[2]+p2[2])/2
    return p3

def distance(p1,p2):
    d1 = (p1[0]-p2[0])**2
    d2 = (p1[1]-p2[1])**2
    d3 = (p1[2]-p2[2])**2

    d = math.sqrt(d1+d2+d3)

    return d

def get_numpy_image(image, camera):
    np_image = np.zeros((camera.H, camera.W, 3))
    pixel_map = image.load()

    for i in range(camera.H):
        for j in range(camera.W):
            np_image[j,i] = np.array(pixel_map[i,j])/255

    return np_image

def get_line(v1, v2, image):
    """
    v1 = (x,y) format
    """
    # print(f'v1 = {v1}, v2 = {v2}')
    # plot(image) #######good to know verticies

    line_image = image

    # if check_boundries(v1,v2):
    #     img1 = ImageDraw.Draw(line_image)  
    #     shape = [v1, v2]
    #     img1.line(shape, fill ="red", width = 0)

    img1 = ImageDraw.Draw(line_image)  
    shape = [v1, v2]
    img1.line(shape, fill ="red", width = 0)


    return line_image

def check_boundries(v1,v2):
    v1_ok = True
    v2_ok = True
    if v1[0]<0 or v1[1]<0: v1_ok = False
    if v2[0]<0 or v2[1]<0: v2_ok = False

    if v1_ok and v2_ok: return True
    else: return False


def get_bb_draw(bb_vertices, room_image):
    """receive a list of 8 vertices in pixel coordinate and returns the bounding box"""
    line_image = room_image
    # plot(line_image, 'get_bb_draw')
        
    line_image = get_line(bb_vertices[0], bb_vertices[1], line_image)
    line_image = get_line(bb_vertices[0], bb_vertices[2], line_image)
    line_image = get_line(bb_vertices[0], bb_vertices[3], line_image)

    # line_image = get_line(bb_vertices[1], bb_vertices[0], line_image)
    line_image = get_line(bb_vertices[1], bb_vertices[6], line_image)
    line_image = get_line(bb_vertices[1], bb_vertices[7], line_image)####### 1,7

    line_image = get_line(bb_vertices[2], bb_vertices[5], line_image)
    line_image = get_line(bb_vertices[2], bb_vertices[7], line_image)
    # line_image = get_line(bb_vertices[2], bb_vertices[0], line_image)
    
    # line_image = get_line(bb_vertices[3], bb_vertices[0], line_image)
    line_image = get_line(bb_vertices[3], bb_vertices[6], line_image)
    line_image = get_line(bb_vertices[3], bb_vertices[5], line_image)

    line_image = get_line(bb_vertices[4], bb_vertices[5], line_image)
    line_image = get_line(bb_vertices[4], bb_vertices[7], line_image)
    line_image = get_line(bb_vertices[4], bb_vertices[6], line_image)
    
    # line_image = get_line(bb_vertices[5], bb_vertices[2], line_image)
    # line_image = get_line(bb_vertices[5], bb_vertices[3], line_image)
    # line_image = get_line(bb_vertices[5], bb_vertices[6], line_image)

    # line_image = get_line(bb_vertices[6], bb_vertices[1], line_image)
    # line_image = get_line(bb_vertices[6], bb_vertices[3], line_image)
    # line_image = get_line(bb_vertices[6], bb_vertices[4], line_image)

    # line_image = get_line(bb_vertices[7], bb_vertices[1], line_image)
    # line_image = get_line(bb_vertices[7], bb_vertices[2], line_image)
    # line_image = get_line(bb_vertices[7], bb_vertices[4], line_image)
    
    # # line_image = get_line(bb_vertices[1], bb_vertices[3], line_image)
    # line_image = get_line(bb_vertices[2], bb_vertices[3], line_image)
    # line_image = get_line(bb_vertices[4], bb_vertices[5], line_image)
    # # line_image = get_line(bb_vertices[5], bb_vertices[6], line_image)#######
    # line_image = get_line(bb_vertices[5], bb_vertices[7], line_image)
    # line_image = get_line(bb_vertices[6], bb_vertices[7], line_image)
    # # line_image = get_line(bb_vertices[0], bb_vertices[4], line_image)#######
    # # line_image = get_line(bb_vertices[1], bb_vertices[5], line_image)#######
    # # line_image = get_line(bb_vertices[2], bb_vertices[6], line_image)#######
    # # line_image = get_line(bb_vertices[3], bb_vertices[7], line_image)#######

    return line_image
    



def show_comparation(default, line_image):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(default)
    plt.subplot(1, 2, 2)
    plt.imshow(line_image)
    plt.show()

if __name__ == "__main__":
    image = load_image('database_organized\database_organized_Area1\conferenceRoom_1\color\camera_874b1dfd225c45dd9fc79b1414c44ca5_conferenceRoom_1_frame_55_domain_rgb.png')
    plot(image)
