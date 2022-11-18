import matplotlib.pyplot as plt
import numpy as np
import cv2

def point2linear_distance(m,n,p):
    return abs(m*p[0]-p[1]+n)/(m**2+1)**(1/2)

def depart_points_linear(points, r_m, r_n, l_m, l_n):

    r_list=[]
    l_list=[]
    
    for p in points:
        if point2linear_distance(r_m, r_n, p) < point2linear_distance(l_m, l_n, p):
            r_list.append(p)
        else:
            l_list.append(p)
            
    return r_list, l_list

def isRedValueable(points, r_m, r_n, l_m, l_n):
    if len(points)<4:
        return False

    r_list, l_list = depart_points_linear(points, r_m, r_n, l_m, l_n)

    print(len(r_list), len(l_list))
    
    if len(r_list)>=2 and len(l_list)>=2:
        return True
    else:
        return False

def findStopArea(img, r_mid, l_mid, cross, p_r_n, p_r_m, p_l_n, p_l_m, isInStopArea):

    r_max=[0,0]
    l_max=[0,0]
    
    if len(r_mid)>0:
        r_max=max(r_mid, key=lambda r:r[1])
    if len(l_mid)>0:
        l_max=max(l_mid, key=lambda l:l[1])

    if r_max==l_max or len(cross)==0:
        return img, r_max[1]
    if r_max[1]>l_max[1]:
        l_max[1]=r_max[1]
        print(p_l_m)
        l_max[0]=int((l_max[1]-p_l_n)/p_l_m)
    else:
        r_max[1]=l_max[1]
        r_max[0]=int((r_max[1]-p_r_n)/p_r_m)
    vertices = np.array([cross, r_max, l_max])
    overlay=img.copy()
    alpha = 0.8
    if isInStopArea==True:
        cv2.fillPoly(overlay, np.int32([vertices]), color=(0,0,255))
    else:
        cv2.fillPoly(overlay, np.int32([vertices]), color=(0,255,0))
    img=cv2.addWeighted(overlay, 0.5, img, 0.5, 0.)
    return img, r_max[1]

def findStopLine(s_mid):
        return min(s_mid, key=lambda s:s[1])[1]

    