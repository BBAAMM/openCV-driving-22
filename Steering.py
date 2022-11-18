import math
import numpy as np

p_r_m=0
p_r_n=0
p_l_m=0
p_l_n=0

outer = 90

def tanh(x):
    y = math.tanh(x)
    return y

def matching(x,input_min,input_max,output_min,output_max):
    return (x-input_min)*(output_max-output_min)/(input_max-input_min)+output_min #map()함수 정의.

def w1(x):
    min = -4
    max = 1.6
    return matching(tanh(matching(x,0,90,min,max)),tanh(min),tanh(max),0,90)

def w2(x):
    min = -4
    max = 1.6
    return matching(tanh(matching(x,180,90,min,max)),tanh(min),tanh(max),180,90)
    
#import time
# for i in range(1, 91):
#     print(i, w(i))
# time.sleep(1)

def w(x):
    if x>=90:
        return w2(x)
    else:
        return w1(x)


def angle_steering(x):
    angle = np.abs(std - x)
    return angle

def initial(x):
    global std
    std = x

def imu_steering(x):
    return matching(angle_steering(x), 0, 90, 90,0)

def speed2angle(a):
    s = 40
    if s == 0:
        return a
    else:
        if a < 90:
            angle = 90 - ((90 - a) * ((100-s)/100))
            return angle
        elif a > 90:
            angle = 90 + ((a - 90) * ((100-s)/100))
            return angle
        else:
            print("speed2angle error")

def outer_control(x):
    global outer
    if 45 > x or x > 135 :
        print('######### 절대 이상치 처리 ############')
    else:
        if outer - 20 < x < outer + 20:
            outer = x
        else:
            print('######### 상대 이상치 처리 ############')
    return outer

def steering_theta(w1, w2):
    if np.abs(w1) > np.abs(w2):  # 우회전
        if w1 * w2 < 0:  #정방향 or 약간 틀어진 방향
            w1 = -w1
            angle = np.arctan(np.abs(math.tan(w1) - math.tan(w2)) / (1 + math.tan(w1)*math.tan(w2)))
            theta = matching(angle, 0, np.pi/2, 90, 180)
        elif w1 * w2 > 0:  #극한으로 틀어진 방향
            if w1 > w2:
                theta = 90
            else:
                theta = 90
        else:
            theta = 0
    elif np.abs(w1) < np.abs(w2) :  # 좌회전
        if w1 * w2 < 0:  #정방향 or 약간 틀어진 방향
            w1 = -w1
            angle = np.arctan(np.abs(math.tan(w1) - math.tan(w2)) / (1 + math.tan(w1)*math.tan(w2)))
            theta = matching(angle, 0, np.pi/2, 90, 0)
        elif w1 * w2 > 0:  #극한으로 틀어진 방향
            if w1 > w2:
                theta = 90
            else:
                theta = 90
        else:
            theta = 0
    else:
        theta = 90

    return theta

def steering_vanishing_point(x, width):
    standard_x = int(width/2)
    diff = standard_x - x 
    if diff >= 0:   #좌회전
        steering_theta = matching(diff, 0, width/2, 90, 45)
    elif diff < 0:
        steering_theta = matching(diff, 0, -width/2, 90, 135)

    return steering_theta
    


# print(w(85))

# print(speed2angle(w(80)))

#initial(180)
# while True:
#     x = 165
#     if std > x: #우
#         print(speed2angle(180 - w(imu_steering(x))))
#     elif x > std:
#         print(speed2angle(w(imu_steering(x))))
#     else:
#         print(90)