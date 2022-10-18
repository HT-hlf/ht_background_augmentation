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
dataset_image_path=r'F:\doing\Motor_detection\datasets_second_remove_noise_add_third\motor_detection_dataset_v2\images\train2017'
dataset_label_path=r'F:\doing\Motor_detection\datasets_second_remove_noise_add_third\motor_detection_dataset_v2\labels\train2017'
dataset_image_list=os.listdir(dataset_image_path)

#保存路径
save_path=r'F:\doing\Motor_detection_dataset\third_dataset\labeled/dataset_dark_noise_2'
save_path_image=os.path.join(save_path,'image')
save_path_label=os.path.join(save_path,'label')
creat_dir(save_path_image)
creat_dir(save_path_label)


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

# 均值滤波
def image_blur(image):
    """
    图像卷积操作：设置卷积核大小，步距
    :param image_path:
    :return:
    """

    # 模糊操作（类似卷积），第二个参数ksize是设置模糊内核大小
    result = cv2.blur(image, (10, 10))
    return result


# 高斯滤波
def image_bifilter(image):
    """
    图像卷积操作：设置卷积核大小，步距
    :param image_path:
    :return:
    """

    # 模糊操作（类似卷积），第二个参数ksize是设置模糊内核大小
    result = cv2.GaussianBlur(image, (0, 0), 3)
    return result

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

# def to_old(image):
#     # print("复古滤镜")
#     height = image.shape[0]
#     width = image.shape[1]
#     channels = image.shape[2]
#     # print("channels : ", channels);
#     print(image.shape)
#     for row in range(height):
#         for col in range(width):
#             blue = image[row, col, 0]
#             green = image[row, col, 1];
#             red = image[row, col, 2]
#             r = (0.393*red + 0.669*green + 0.049*blue);
#             g = (0.349*red + 0.686*green + 0.048*blue);
#             b = (0.272*red + 0.534*green + 0.011*blue);
#             image[row, col, 0] = b;
#             image[row, col, 1] = g;
#             image[row, col, 2] = r;
#     return image

def to_old(img):
    rows, cols = img.shape[:2]

    # 新建目标图像
    dst = np.zeros((rows, cols, 3), dtype="uint8")

    # 图像怀旧特效
    for i in range(rows):
        for j in range(cols):
            B = 0.272 * img[i, j][2] + 0.534 * img[i, j][1] + 0.131 * img[i, j][0]
            G = 0.349 * img[i, j][2] + 0.686 * img[i, j][1] + 0.168 * img[i, j][0]
            R = 0.393 * img[i, j][2] + 0.769 * img[i, j][1] + 0.189 * img[i, j][0]
            if B > 255:
                B = 255
            if G > 255:
                G = 255
            if R > 255:
                R = 255
            dst[i, j] = np.uint8((B, G, R))

    return dst

def maoboli(src):
    dst = np.zeros_like(src)

    # 获取图像行和列
    rows, cols = src.shape[:2]

    # 定义偏移量和随机数
    offsets = 10
    random_num = 0

    # 毛玻璃效果: 像素点邻域内随机像素点的颜色替代当前像素点的颜色
    for y in range(rows - offsets):
        for x in range(cols - offsets):
            random_num = np.random.randint(0, offsets)
            dst[y, x] = src[y + random_num, x + random_num]
    return dst

def get_resize_num(img1,img2):
    img1_height, img1_width,_ = img1.shape
    img2_height, img2_width,_ = img2.shape
    if (img1_width/img1_height>img2_width/img2_height):
        return (img1_width,int(img2_height/img2_width*img1_width))
    else:
        return (int(img2_width/img2_height*img1_height),img1_height)
dataset_image_list=dataset_image_list[::-1]
for i in range(len(dataset_image_list)):
    dataset_image = dataset_image_list[i]
    dataset_label = dataset_image.rstrip('.jpg') + '.txt'
    aug_dataset_image = 'aug_dark_noise_1_' + dataset_image
    aug_dataset_label = 'aug_dark_noise_1_' + dataset_label
    dataset_image_abs = dataset_image_path + '/' + dataset_image
    dataset_label_abs = dataset_label_path + '/' + dataset_label
    print('----process: {}-----'.format(dataset_image_abs))
    save_image= save_path_image +'/' + aug_dataset_image

    save_label = save_path_label +'/'+ aug_dataset_label

    shutil.copy(dataset_label_abs, save_label)

    dataset_image_cv = cv2.imread(dataset_image_abs)
    # cv2.imshow('raw', dataset_image_cv)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imshow('dataset', dataset_image_cv)
    random_count=random.randint(0,5)
    print(random_count)
    # random_count = 5
    if random_count==0:
        dataset_image_cv=SaltAndPepper(dataset_image_cv,random.uniform(0.01,0.10))
    elif random_count==1:
        dataset_image_cv =  addGaussianNoise(dataset_image_cv,random.uniform(0.01,0.10))
    elif random_count==2:
        dataset_image_cv =  image_blur(dataset_image_cv)
    elif random_count==3:
        dataset_image_cv =  image_bifilter(dataset_image_cv)
    elif random_count==4:
        dataset_image_cv =  to_old(dataset_image_cv)
    elif random_count==5:
        dataset_image_cv =  maoboli(dataset_image_cv)
    else:
        pass


    if random.randint(0, 1):
        dataset_image_cv = darker(dataset_image_cv,random.uniform(0.75,0.98))
    else:
        dataset_image_cv = brighter(dataset_image_cv, random.uniform(1.05,1.2))





    cv2.imwrite(save_image,dataset_image_cv)
    # cv2.imshow('dataset_aug',dataset_image_cv)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
print('Finish.')




