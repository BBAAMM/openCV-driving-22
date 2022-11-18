import cv2
import numpy as np


def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def threshold(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.threshold(img, low_threshold, high_threshold, cv2.THRESH_BINARY)

def region_of_interest(img, vertices):
    mask=np.zeros_like(img)
    
    if len(img.shape)>2:
        channel_count = img.shape[2]
        ignore_mask_color=(255,)*channel_count
    else:
        ignore_mask_color=255

    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_image=cv2.bitwise_and(img,mask)
    return masked_image

def preprocessing(img, low_threshold, high_threshold, kernel_size): #640*360
    
    img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    if len(img.shape)>2:
        channel_count=img.shape[2]
        ignore_mask_color=(255,) * channel_count
    else:
        ignore_mask_color=255

    mask=np.zeros_like(img)
    vertices=np.array([[(20, 315),
                        (20, 210),
                       (160, 130),
                       (470, 130),
                        (620, 210),
                       (620, 315)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image=region_of_interest(img_hsv, vertices)

    lower_blue = (110-3, 120, 150) # hsv 이미지에서 바이너리 이미지로 생성 , 적당한 값 30
    upper_blue = (110+5, 255, 255)

    lower_red = (6-6, 110, 120) # hsv 이미지에서 바이너리 이미지로 생성 , 적당한 값 30
    upper_red = (6+4, 255, 255)

    lower_yellow = (19-1, 110, 120) # hsv 이미지에서 바이너리 이미지로 생성 , 적당한 값 30
    upper_yellow = (19+5, 255, 255)

    mask_hsv_red = cv2.inRange(masked_image, lower_red, upper_red)
    mask_hsv_blue = cv2.inRange(masked_image, lower_blue, upper_blue)
    mask_hsv_yellow = cv2.inRange(masked_image, lower_yellow, upper_yellow)

    mask_hsv=cv2.bitwise_or(mask_hsv_red, mask_hsv_blue)
    mask_hsv=cv2.bitwise_or(mask_hsv, mask_hsv_yellow)
    
    stop_img = cv2.bitwise_and(img, img, mask=mask_hsv)
    # img_gray=grayscale(mask_hsv)
    img_blur = gaussian_blur(mask_hsv, kernel_size)
    ret, img_thresh=threshold(img_blur, low_threshold, high_threshold)
    return img_thresh
