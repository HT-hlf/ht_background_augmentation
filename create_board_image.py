# coding:utf-8
# @Author     : HT
# @Time       : 2022/9/29 10:08
# @File       : create_board_image.py
# @Software   : PyCharm


import numpy as np
import cv2 as cv

# img1 = np.zeros((640,480,3),np.uint8)
#指定尺寸大小
img1 = np.zeros((1280,960,3),np.uint8)
# 输入想要颜色的BGR，创建纯色图
img1[:] = [0,255,0]

cv.namedWindow('img1', cv.WINDOW_AUTOSIZE)
cv.imshow('img1',img1)
cv.imwrite('green_image.jpg',img1)
cv.waitKey(0)
cv.destroyAllWindows()
