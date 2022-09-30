# coding:utf-8
# @Author     : HT
# @Time       : 2022/9/29 10:19
# @File       : background_augmentation.py
# @Software   : PyCharm


#无太多修改，较上版
import os
import cv2
import random
import numpy as np
import shutil
def creat_dir(out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)


# 数据集路径
dataset_image_path=r'G:\doing\Motor_detection_dataset\second_dataset\labeled\zhuohang3_8_green\image'
dataset_image_list=os.listdir(dataset_image_path)
# 背景路径
background_path=r'G:\doing\Motor_detection_dataset\second_dataset\labeled\background'
background_path_list=os.listdir(background_path)
#保存路径
save_path=r'G:\doing\Motor_detection_dataset\second_dataset\labeled/test'
creat_dir(save_path)

# lower_green=np.array([58,253,80])
# upper_green=np.array([62,257,90])

# lower_green=np.array([48,200,40])
# upper_green=np.array([62,257,90])
#绿色
# lower_green=np.array([35,43,46])
# upper_green=np.array([77,255,255])
#绿色+青色
# lower_green=np.array([35,43,46])
# upper_green=np.array([99,255,255])

# lower_green=np.array([60,210,46])
# upper_green=np.array([75,250,255])

lower_green=np.array([60,190,46])
upper_green=np.array([75,255,255])

# 黑白色翻转（我记得opencv是有现成函数来着）
# def invert(src_img):
#     height, width=src_img.shape
#     dst_img = np.zeros((height, width), np.uint8)
#     for i in range(0, height):
#         for j in range(0, width):
#             grayPixel = src_img[i, j]
#             dst_img[i, j] = 255 - grayPixel
#     return dst_img


def get_resize_num(img1,img2):
    img1_height, img1_width,_ = img1.shape
    img2_height, img2_width,_ = img2.shape
    if (img1_width/img1_height>img2_width/img2_height):
        return (img1_width,int(img2_height/img2_width*img1_width))
    else:
        return (int(img2_width/img2_height*img1_height),img1_height)

for i in range(len(dataset_image_list)):
    dataset_image = dataset_image_list[i]
    dataset_image_abs = dataset_image_path + '/' + dataset_image
    print('----process: {}-----'.format(dataset_image_abs))
    save_image= save_path +'/' + dataset_image

    dataset_image_cv = cv2.imread(dataset_image_abs)
    dataset_image_cv_hsv = cv2.cvtColor(dataset_image_cv, cv2.COLOR_BGR2HSV)
    # dataset_image_height,dataset_image_width=dataset_image_cv.shape

    background_image=background_path_list[random.randint(0,len(background_path_list)-1)]
    background_image_abs = background_path + '/' + background_image
    background_image_abs_cv = cv2.imread(background_image_abs)
    # background_image_height, background_image_width = background_image_abs_cv.shape


    background_image_resize=cv2.resize(background_image_abs_cv,get_resize_num(dataset_image_cv, background_image_abs_cv))




    mask = cv2.inRange(dataset_image_cv_hsv, lower_green, upper_green)

    #膨胀
    mask = cv2.dilate(mask, (3, 3), iterations=7)
    # # cv2.imshow('dilate', mask)
    # #腐蚀
    mask = cv2.erode(mask, (3,3), iterations=1)
    # # cv2.imshow('erode', mask)


    # kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # erode = cv2.morphologyEx(mask_invert, cv2.MORPH_ERODE,kernel=kernel1)

    height,width=mask.shape
    for i in range(height):
        for j in range(width):
            if mask[i, j] == 255:
                dataset_image_cv[i,j, 0:3] = background_image_resize[i, j, 0:3]

    cv2.imwrite(save_image,dataset_image_cv)
    # cv2.imshow('dataset',dataset_image_cv)
print('Finish.')

    # cv2.imshow('mask', mask)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()




