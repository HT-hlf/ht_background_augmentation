# coding:utf-8
# @Author     : HT
# @Time       : 2022/9/29 10:19
# @File       : background_augmentation.py
# @Software   : PyCharm


import os
import cv2
import random
import numpy as np
import shutil
def creat_dir(out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)


# 数据集路径
dataset_image_path=r'G:\doing\Motor_detection_dataset\second_dataset\labeled\second_yolo_augementation\images'
dataset_label_path=r'G:\doing\Motor_detection_dataset\second_dataset\labeled\second_yolo_augementation\labels'
dataset_image_list=os.listdir(dataset_image_path)
# 背景路径
background_path=r'G:\doing\Motor_detection_dataset\second_dataset\labeled\background1'
background_path_list=os.listdir(background_path)
#保存路径
save_path=r'G:\doing\Motor_detection_dataset\second_dataset\labeled/test_dataset'
save_path_image=os.path.join(save_path,'image')
save_path_label=os.path.join(save_path,'label')
creat_dir(save_path_image)
creat_dir(save_path_label)

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

lower_green=np.array([60,220,46])
upper_green=np.array([84,255,255])

# 黑白色翻转（我记得opencv是有现成函数来着）
# def invert(src_img):
#     height, width=src_img.shape
#     dst_img = np.zeros((height, width), np.uint8)
#     for i in range(0, height):
#         for j in range(0, width):
#             grayPixel = src_img[i, j]
#             dst_img[i, j] = 255 - grayPixel
#     return dst_img


# 椒盐噪声
def SaltAndPepper(src, percetage):
    SP_NoiseImg = src.copy()
    SP_NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(SP_NoiseNum):
        randR = np.random.randint(0, src.shape[0] - 1)
        randG = np.random.randint(0, src.shape[1] - 1)
        randB = np.random.randint(0, 3)
        if np.random.randint(0, 1) == 0:
            SP_NoiseImg[randR, randG, randB] = 0
        else:
            SP_NoiseImg[randR, randG, randB] = 255
    return SP_NoiseImg


# 高斯噪声
def addGaussianNoise(image, percetage):
    G_Noiseimg = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    G_NoiseNum = int(percetage * image.shape[0] * image.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(0, h)
        temp_y = np.random.randint(0, w)
        G_Noiseimg[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0]
    return G_Noiseimg


# 昏暗
def darker(image, percetage=0.9):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get darker
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = int(image[xj, xi, 0] * percetage)
            image_copy[xj, xi, 1] = int(image[xj, xi, 1] * percetage)
            image_copy[xj, xi, 2] = int(image[xj, xi, 2] * percetage)
    return image_copy


# 亮度
def brighter(image, percetage=1.5):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get brighter
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = np.clip(int(image[xj, xi, 0] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 1] = np.clip(int(image[xj, xi, 1] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 2] = np.clip(int(image[xj, xi, 2] * percetage), a_max=255, a_min=0)
    return image_copy



def get_resize_num(img1,img2):
    img1_height, img1_width,_ = img1.shape
    img2_height, img2_width,_ = img2.shape
    if (img1_width/img1_height>img2_width/img2_height):
        return (img1_width,int(img2_height/img2_width*img1_width))
    else:
        return (int(img2_width/img2_height*img1_height),img1_height)

for i in range(len(dataset_image_list)):
    dataset_image = dataset_image_list[i]
    dataset_label = dataset_image.rstrip('.jpg') + '.txt'
    aug_dataset_image = 'aug_1' + dataset_image
    aug_dataset_label = 'aug_1' + dataset_label
    dataset_image_abs = dataset_image_path + '/' + dataset_image
    dataset_label_abs = dataset_label_path + '/' + dataset_label
    print('----process: {}-----'.format(dataset_image_abs))
    save_image= save_path_image +'/' + aug_dataset_image

    save_label = save_path_label +'/'+ aug_dataset_label

    shutil.copy(dataset_label_abs, save_label)







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
    # mask = cv2.dilate(mask, (3, 3), iterations=10)
    # # cv2.imshow('dilate', mask)
    # #腐蚀
    # mask = cv2.erode(mask, (3,3), iterations=1)
    # # cv2.imshow('erode', mask)


    # kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # erode = cv2.morphologyEx(mask_invert, cv2.MORPH_ERODE,kernel=kernel1)

    height,width=mask.shape
    for i in range(height):
        for j in range(width):
            if mask[i, j] == 255:
                dataset_image_cv[i,j, 0:3] = background_image_resize[i, j, 0:3]

    # cv2.imshow('dataset', dataset_image_cv)
    if random.randint(0,1):
        dataset_image_cv=SaltAndPepper(dataset_image_cv,random.uniform(0.01,0.15))
    else:
        dataset_image_cv =  addGaussianNoise(dataset_image_cv,random.uniform(0.01,0.1))
    if random.randint(0, 1):
        dataset_image_cv = darker(dataset_image_cv,random.uniform(0.75,0.98))
    else:
        dataset_image_cv = brighter(dataset_image_cv, random.uniform(1.05,1.2))





    cv2.imwrite(save_image,dataset_image_cv)
    # cv2.imshow('dataset_aug',dataset_image_cv)


    # cv2.imshow('mask', mask)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
print('Finish.')




