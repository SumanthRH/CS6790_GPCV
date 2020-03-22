import numpy as np 
import cv2
from scipy.linalg import inv, cholesky,svd, null_space 
import matplotlib.pyplot as plt
DEBUG = True

def construct2rows(list_):
        x1,y1,x2,y2 = list_
        x1,y1,x2,y2 = x1, y1, x2, y2
        row1 = np.array([x1,y1,1,0,0,0,-1*x1*x2,-1*y1*x2, -x2])
        row2 = np.array([0,0,0,x1,y1,1,-1*x1*y2,-1*y1*y2, -y2])
        return np.stack([row1,row2])

def point_corr_H(point_list):
        A = [construct2rows(list_) for list_ in point_list]
        A = np.concatenate(A).astype('float64')
        print(point_list[:,2:])
#         b = point_list[:,2:].copy()
#         b = b.flatten().astype(float)
        u,d,vh = svd(A)
#         print('The A matrix for H :',A)
#         b[::2] = b[::2]/width
#         b[1::2] =  b[1::2]/height
#         print(' b, normalized',b)
        return vh[-1]

def construct_grid(width, height, homogenous):
        coords = (np.indices((width, height)).reshape(2, -1)).astype(float)
        coords[0] = coords[0]
        coords[1] = coords[1]
        return np.vstack((coords, np.ones(coords.shape[1]))).astype(np.int) if homogenous else coords

def get_H(point_list):
        H = point_corr_H(point_list)
#         print("Homography estimate: ",H)
#         H = H.tolist()
#         H.append(1)
        H = np.array(H,dtype='float64').reshape(3,3)
        if H[2,2]:
            H = H/H[2,2]
        return H



def get_line(point1,point2):
    point1_,point2_ = point1.copy(),point2.copy()
    if len(point1)==2:
        point1_ = point1_.tolist()
        point1_.append(1)
        point1_ = np.array(point1_)
        point2_ = point2_.tolist()
        point2_.append(1)
        point2_ = np.array(point2_)
    assert len(point1_)==3 and len(point2_)==3
    line = np.cross(point1_,point2_)
    if line[-1]:
        line = line/line[-1]
    return line
    # lines = [line/line[-1] for line in lines if line[-1] else line]


def get_l_inf(lines):
    inf_points = [np.cross(lines[0],lines[1]),np.cross(lines[2],lines[3])]
    l_inf = np.cross(inf_points[0],inf_points[1])
    if l_inf[-1]:
        l_inf = l_inf/l_inf[-1]
    return l_inf

def get_transf_point(points,H):
    transf_points = np.matmul(H,points.T).T 
    for i in range(len(transf_points)):
        if transf_points[i][-1]:
            transf_points[i] = transf_points[i]/transf_points[i][-1]
    return transf_points 


def construct_P(lines1,lines2):
    row1 = np.array([lines1[0][0]*lines1[1][0],lines1[1][0]*lines1[0][1] + lines1[1][1]*lines1[0][0],\
                    lines1[1][1]*lines1[0][1]])
    row2 = np.array([lines2[0][0]*lines2[1][0],lines2[1][0]*lines2[0][1] + lines2[1][1]*lines2[0][0],\
                    lines2[1][1]*lines2[0][1]])
    P = np.stack([row1,row2])
#     u,d,vh =  svd(P)
#     if DEBUG:
#         print(P)
        
    return null_space(P)
