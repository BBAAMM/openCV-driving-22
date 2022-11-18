from tracemalloc import start
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time

from Steering import *
from Preprocessing import *
from StopDetector import *

WIDTH  = 640
HEIGHT = 360

kernel_size=11

low_threshold=120
high_threshold=255

theta=np.pi/180

lower_blue = (115-30, 10, 10) # hsv 이미지에서 바이너리 이미지로 생성 , 적당한 값 30
upper_blue = (115+30, 255, 255)

lower_red = (6-6, 30, 30) # hsv 이미지에서 바이너리 이미지로 생성 , 적당한 값 30
upper_red = (6+4, 255, 255)

lower_yellow = (19-4, 30, 30) # hsv 이미지에서 바이너리 이미지로 생성 , 적당한 값 30
upper_yellow = (19+30, 255, 255)

isUseableRed=False
isInStopArea=False

def setup_path():
    path = "./source/KakaoTalk_20221007_063507621.mp4"
    cap=cv2.VideoCapture(path) #path
    return cap

def setup_countours():
    obj_b = cv2.imread('./source/corn_data/lavacorn_nb.png', cv2.IMREAD_GRAYSCALE)#wad
    obj_s = cv2.imread('./source/corn_data/lavacorn_ns.png', cv2.IMREAD_GRAYSCALE)#wad
    obj_contours_b,_=cv2.findContours(obj_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)#wad
    obj_contours_s,_=cv2.findContours(obj_s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)#wad
    obj_pts_b=obj_contours_b[0]#wad
    obj_pts_s=obj_contours_s[0]#wad

    return obj_pts_b, obj_pts_s

def setup_output(_path, _cap):
    fps = _cap.get(cv2.CAP_PROP_FPS)
    codec = cv2.VideoWriter_fourcc(*'DIVX')
    out=cv2.VideoWriter(_path, codec, fps, (int(WIDTH),int(HEIGHT)))

    return out

def setup_linear_reg():
    global p_r_m
    global p_r_n
    global p_l_m
    global p_l_n

    p_r_m=0.3
    p_r_n=37
    p_l_m=-0.3
    p_l_n=238

def depart_points(img, points):
    right_line=[]
    left_line=[]
    stop_line=[]

    for p in points:
        x1, y1, x2, y2 = p
        label = plot_one_box([x1, y1, x2, y2], img)

        x = int((x1 + x2)/2)
        y = int(y2)

        if label == 'blue':
            left_line.append([x, y])
        elif label == 'yellow':
            right_line.append([x, y])
        elif label == 'red':
            stop_line.append([x, y])
        else:
            pass

    return right_line, left_line, stop_line

def hsv_inrange(h,s,v):
    if lower_blue[0] <= h <= upper_blue[0] and lower_blue[1] <=s<= upper_blue[1] and lower_blue[2] <=v<= upper_blue[2]:
        color = (255,0,0)
        label = 'blue'
    elif lower_yellow[0] <= h <= upper_yellow[0] and lower_yellow[1] <=s<= upper_yellow[1] and lower_yellow[2] <=v<= upper_yellow[2]:
        color = (0,255,255)
        label = 'yellow'
    elif lower_red[0] <= h <= upper_red[0] and lower_red[1] <=s<= upper_red[1] and lower_red[2] <=v<= upper_red[2]:
        color = (0,0,255)
        label = 'red'
    else:
        color = (0,0,0)
        label = 'another'
        
    return label, color

def plot_one_box(x, img, line_thickness=None):  # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))

    colr = np.mean(img[(c1[1]+c2[1])//2:c2[1], int(0.5 * (c1[0] + c2[0])), :],axis=0)
    
    b,g,r = colr
    
    color = [b,g,r]  # BGR 순서 ; 파란색
    pixel = np.uint8([[color]]) # 한 픽셀로 구성된 이미지로 변환

    # BGR -> HSV
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)
    # print(hsv, 'shape:', hsv.shape )

    # 픽셀값만 가져오기 
    hsv = hsv[0][0]

    h, s, v = hsv[0], hsv[1], hsv[2]
    b,g,r=colr/np.sum(colr)

    label, color = hsv_inrange(h,s,v)

    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    return label

def gradient(p1, p2):
    return (p2[1]-p1[1])/(p1[0]-p2[0])

def y_intercept(m, p):
    return (-1)*p[1]-m*p[0]

def point2linear_distance(m,n,p):
    return abs(m*p[0]-p[1]+n)/(m**2+1)**(1/2)

def linear_regs(img, left, right, isRedCorn):
    global p_r_m
    global p_r_n
    global p_l_m
    global p_l_n

    left_calculated_weight, left_calculated_bias, target_l = regression(left, p_l_m, p_l_n)
    right_calculated_weight, right_calculated_bias, target_r = regression(right, p_r_m, p_r_n)

    img = cv2.line(img,(int(WIDTH/2),HEIGHT),(int(WIDTH/2),int(0)),(255,0,0),3)

    cross_x = (right_calculated_bias - left_calculated_bias) / (left_calculated_weight - right_calculated_weight)
    cross_y = left_calculated_weight*((right_calculated_bias - left_calculated_bias)/(left_calculated_weight - right_calculated_weight)) + left_calculated_bias

    if np.isnan(cross_x)==False and np.isnan(cross_y)==False:
        if isRedCorn:
            img = cv2.line(img,(0,int(left_calculated_bias)),(int(WIDTH),int(target_l)),(0,0,255), 3)
            img = cv2.line(img,(int(0),int(right_calculated_bias)),(WIDTH,int(target_r)),(0,0,255),3)
        else :
            img = cv2.line(img,(0,int(left_calculated_bias)),(int(WIDTH),int(target_l)),(255,0,0), 3)
            img = cv2.line(img,(int(0),int(right_calculated_bias)),(WIDTH,int(target_r)),(0,255,255),3)
        cv2.circle(img, (int(cross_x), int(cross_y)), 10, (0, 0, 255), -1, cv2.LINE_AA)

        x = steering_theta(left_calculated_weight, right_calculated_weight)
        if 80<x<110:
            print("소실점 조향 서보모터 각도: ", speed2angle(w(outer_control(steering_vanishing_point(cross_x, WIDTH)))))
        else:
            print("기울기 조향 서보모터 각도: ", speed2angle(w(outer_control(x))))

    p_l_m=left_calculated_weight
    p_r_m=right_calculated_weight
    p_l_n=left_calculated_bias
    p_r_n=right_calculated_bias
    #print('Done.')
    return img, [cross_x, cross_y]

def regression(point, p_m, p_n):
    x=[]
    y=[]

    for i in range(len(point)):
        x.append(point[i][0])
        y.append(point[i][1])
    
    calculated_weight=0

    if len(point)<2:
        calculated_weight=p_m
        calculated_bias=p_n
    else:
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        calculated_weight=least_square(x,y,mean_x,mean_y)
        calculated_bias=mean_y-calculated_weight*mean_x
    target=calculated_weight*WIDTH+calculated_bias
    print(f"y = {calculated_weight} * X + {calculated_bias}")

    return calculated_weight, calculated_bias, target

def least_square(val_x, val_y, mean_x, mean_y):
    return ((val_x - mean_x) * (val_y - mean_y)).sum() / ((val_x - mean_x)**2).sum()

    
def find_contours(img_thresh, obj_pts_b, obj_pts_s)->list:
    contours,_=cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    points = []
    isSameCorn=False
    for pts in contours:
        if cv2.contourArea(pts) <60:
            continue
        rc=cv2.boundingRect(pts)
        dist_b=cv2.matchShapes(obj_pts_b, pts, cv2.CONTOURS_MATCH_I3, 0)
        dist_s=cv2.matchShapes(obj_pts_s, pts, cv2.CONTOURS_MATCH_I3, 0)
        if dist_b <0.5 or dist_s<0.4:
            mid_x = (2*rc[0]+rc[2])/2
            for p in points:
                if p[0]<=mid_x and p[2]>=mid_x:
                    isSameCorn=True
                    break
            if not isSameCorn and 40<=mid_x<=600:
                cv2.rectangle(img_thresh, rc, (255, 0,0),1)
                cv2.imshow("img", img_thresh)
                points.append([rc[0], rc[1], rc[0]+rc[2], rc[1]+rc[3]])
            isSameCorn=False

    return points

def draw_circle(img, points, color:tuple):
    for p in points:
        cv2.circle(img, p, 10, color, -1, cv2.LINE_AA)
    
def starter():

    global isUseableRed
    global isInStopArea

    cap = setup_path()
    obj_pts_b, obj_pts_s = setup_countours()
    out = setup_output('./source/output_16_line.mp4', cap)
    setup_linear_reg()

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            print("프레임을 수신할 수 없습니다. 종료 중 ...")
            break
        t=time.time()
        img = cv2.resize(img, dsize=(int(WIDTH), int(HEIGHT)))

        img_thresh = preprocessing(img, low_threshold, high_threshold, kernel_size)

        points = find_contours(img_thresh, obj_pts_b, obj_pts_s)
        r_mid, l_mid, s_mid = depart_points(img, points)
        draw_circle(img, r_mid, (0, 255, 255))#노랑
        draw_circle(img, l_mid, (255, 0, 0)) #파랑
        
        if isUseableRed==False and isRedValueable(s_mid, p_r_m, p_r_n, p_l_m, p_l_n):
            isUseableRed=True

        if isUseableRed==True:
            if not isInStopArea:
                r_mid, l_mid = depart_points_linear(s_mid, p_r_m, p_r_n, p_l_m, p_l_n)
                linear_img, cross=linear_regs(img, l_mid, r_mid, True)
                if p_l_m!=0 and p_r_m!=0:
                    linear_img, y_val = findStopArea(linear_img, r_mid, l_mid, cross, p_r_n, p_r_m, p_l_n, p_l_m, isInStopArea)
                else:
                    continue
            else:
                linear_img = img
            if len(s_mid)!=0:
                stopLine_y = findStopLine(s_mid)
                if stopLine_y!=-1:
                    cv2.line(linear_img, (0, stopLine_y), (int(WIDTH), stopLine_y), (0,255,0), 2)
                if stopLine_y>=230:
                    print("----- 정지선에 도착했습니다 -----")
                    break
            cv2.line(linear_img, (0,230), (int(WIDTH), 230), (0, 0, 255), 2)
            if 230-y_val<=0:
                isInStopArea=True
            if isInStopArea:
                print("----- 정지 영역 -----")
        else:
            linear_img, cross=linear_regs(img, l_mid, r_mid, False)

        dt=time.time()-t
        print("delay : ", dt)
        cv2.imshow("ex", linear_img)#lines_edges
        out.write(linear_img)
        
        if cv2.waitKey(1) == ord('q'):
            break
    # 작업 완료 후 해제
    cap.release()
    out.release()
    cv2.destroyAllWindows()

starter()