from helper import * 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os 
from scipy.linalg import null_space, cholesky, inv, svd
from scipy.interpolate import interp2d
# %matplotlib inline
from helper import *
# get_H()
DEBUG = True

path = './Images/Image1.JPG'
img = cv2.imread(path)
img = cv2.resize(img,(img.shape[1]//10,img.shape[0]//10),fx=4,fy=4)
plt.imshow(img[:,:,::-1])
height, width = img.shape[:2]
print(height,width)

point_list = np.array([[2360,1736,2400,1750],[4416,1200,4525,1750],[3416,3904,2400,4162],[5816,2920,4525,4162]])
point_list = point_list//10
print(point_list)
cv2.imwrite('resized.png',img)
H = get_H(point_list)
print(H)

def blur(arr,arr2,ind,size=3):
    if size%2:
        try :
            seq = [arr[tuple([ind[0]+i,ind[1]+j,ind[2]])] for i in range((-size+1)//2,(size+1)//2) for j in range((-size+1)//2,(size+1)//2)]
            arr2[tuple(ind)] = max(seq) 
#             print('SEQUENCE',seq)
        except IndexError:
#             seq = [arr[tuple([ind[0]+i,ind[1]+j,ind[2]])] for i in range((-size+1)//2,(size-1)//2) for j in range((-size+1)//2,(size-1)//2)]
            pass

def get_arr_from_H(img,width,height,H,size=False,scale=False,buf=50):
    grid = construct_grid(width,height,True)
    if scale :
        diag = np.diag([1/width, 1/height, 1])
        H = inv(diag)@H@diag
    # print(H_perp.shape)
    aff_grid = np.matmul(H,grid)
        
#     aff_grid[0] = aff_grid[0]*width
#     aff_grid[1] = aff_grid[1]*height
    if DEBUG:
        print(aff_grid.shape)
    x_aff = np.round(np.divide(aff_grid[0],aff_grid[2])).astype(int)    
    y_aff = np.round(np.divide(aff_grid[1],aff_grid[2])).astype(int)
    if DEBUG:
        print("X_aff max, min :",max(x_aff),min(x_aff))
        print("y_aff max:",max(y_aff),min(y_aff))
   
    x_aff -= buf
    y_aff -= buf
    if min(x_aff) < buf:
        x_aff -= min(x_aff)
        y_aff -= min(y_aff)
    if DEBUG:
        print("X_aff max, min :",max(x_aff),min(x_aff))
        print("y_aff max:",max(y_aff),min(y_aff))
    a, b = max(x_aff), max(y_aff)
    if not size :
        inds = np.where((x_aff>= 0) & (y_aff>=0) & (x_aff<width) & (y_aff<height))
        aff = np.zeros_like(img)
    else :
        inds = np.where((x_aff>=0) & (y_aff>=0) & (x_aff<max(x_aff)) & (y_aff<max(y_aff)))
        aff = np.zeros(shape=(max(y_aff),max(x_aff),3),dtype=float)
    x_aff,y_aff = x_aff[inds],y_aff[inds]
#     print(x_aff)
    if DEBUG:
        print("X_aff max, min :",max(x_aff),min(x_aff))
        print("y_aff max:",max(y_aff),min(y_aff))
#         print(np.sum(x_aff<1000))
        
    xx,yy =  grid[0][inds], grid[1][inds]
    
    aff[y_aff,x_aff] = img[yy, xx]
    zeros = np.array(np.where(aff==0))
    aff1 = aff.copy()
    nzs = np.where(aff != 0)
    for ind in zeros.T:
        blur(aff,aff1,ind,5)
#         break
#         except Exception as e:
#             print(e)
#             continue

    if DEBUG :
        print(np.max(aff))
    kernel = np.ones((5,5),np.float32)/25

#     aff = cv2.filter2D(aff,-1,kernel)
    
#     aff = cv2.resize(aff,(img.shape[1],img.shape[0]))

    aff1[nzs] = aff[nzs]
    plt.imshow(aff1[:,:,::-1].astype(int))
    return aff1 

# a = cv2.warpPerspective(img,H,(width,height))
a = get_arr_from_H(img, width, height, H,size=False)
plt.imshow(a[:,:,::-1].astype(int))
print(a.shape)
print(point_list)

M = cv2.getPerspectiveTransform(point_list[:,:2].astype(np.float32),point_list[:,2:].astype(np.float32))
print(M)
print(H)
scale = np.diag([1/width, 1/height, 1])
warped = cv2.warpPerspective(img,H,(width,height))
plt.imshow(warped[:,:,::-1])

lines = np.stack([get_line(points[0],points[1]),get_line(points[2],points[3]),\
                    get_line(points[0],points[2]),get_line(points[1],points[3])])
l_inf = get_l_inf(lines)
H_perp = np.array([[1,0,0],[0,1,0],l_inf])

lines_perp = get_transf_line(lines,H_perp)
# print(lines_perp[:2])
try :
    lines_perp = np.array([line/line[-1] for line in lines_perp])
    print(lines_perp)
except Exception:
    pass


s = construct_P(lines_perp[[0,2]],lines_perp[[1,2]])
if len(s.shape)==2:
    if np.sign(s[0][0]*s[2][0]) ==1:
        s= s[:,0]
    else :
        s = s[:,1]

kkt = np.array([[s[0],s[1]],[s[1],s[2]]]).reshape(2,2)
u,d,_ = svd(kkt)

print(u,d)
k = np.matmul(u,np.diag(np.sqrt(d)))
k = k* np.sign(k[0,0])
transformed = (H_perp@points.T).T
print(k)
transformed = [point/point[-1] for point in transformed]
# print(transformed)
print(get_line(transformed[2],transformed[0]))
