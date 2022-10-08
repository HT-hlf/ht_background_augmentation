# coding:utf-8
# @Author     : HT
# @Time       : 2022/9/30 8:15
# @File       : get_hsv.py
# @Software   : PyCharm

#encoding=UTF-8

import cv2
# 读取图片
img = cv2.imread(r'G:\doing\Motor_detection_dataset\second_dataset\labeled\second_yolo_augementation\images/310n_21.jpg')  # 直接读为灰度图像
height, width = img.shape[:2]
# 为了方便显示，可以根据图片大小，进行缩放
# size = (int(width * 0.5), int(height * 0.5))  # bgr
size = (int(width * 0.25), int(height * 0.25))  # bgr
img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

#BGR转化为HSV
HSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#鼠标点击响应事件
def getposHsv(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN:
        print("HSV is",HSV[y,x])
def getposBgr(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN:
        print("Bgr is",img[y,x])
#
cv2.imshow("imageHSV",HSV)
cv2.imshow('image',img)
cv2.setMouseCallback("imageHSV",getposHsv)
cv2.setMouseCallback("image",getposBgr)
cv2.waitKey(0)