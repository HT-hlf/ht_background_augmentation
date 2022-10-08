import numpy as np
import cv2
import math

def flip(img, label, width):
    dst = cv2.flip(img, 1)
    label_n = list(label)
    label_n[0] = width-1-label[2]
    label_n[2] = width-1-label[0]
    return dst, label_n


def gamma_transformation(img, label):
    img_float = np.float32(img)
    img_norm = img_float / 255.0
    gamma = (1.1-0.8)*np.random.random_sample() + 0.8
    # print "gamma: ", gamma
    dst = np.power(img_norm, gamma)
    matrix_255 = 255*np.ones_like(dst)
    dst = np.uint8(dst * matrix_255)
    return dst, label

def add_sub_pixels(img, label):
    dst = np.float32(img)
    matrix_values = (np.random.randint(-15,15))*np.ones_like(img)
    # matrix_values = -80 * np.ones_like(img)
    dst = np.minimum(np.maximum(dst+matrix_values,0), 255)
    dst = np.uint8(dst)
    return dst, label


def saltpepper(img,n):
    m=int((img.shape[0]*img.shape[1])*n)
    for a in range(m):
        i=int(np.random.random()*img.shape[1])
        j=int(np.random.random()*img.shape[0])
        if img.ndim==2:
            img[j,i]=np.random.randint(155,255)
        elif img.ndim==3:
            img[j,i,0]=np.random.randint(155,255)
            img[j,i,1]=np.random.randint(155,255)
            img[j,i,2]=np.random.randint(155,255)
    for b in range(m):
        i=int(np.random.random()*img.shape[1])
        j=int(np.random.random()*img.shape[0])
        if img.ndim==2:
            img[j,i]=np.random.randint(0,100)
        elif img.ndim==3:
            img[j,i,0]=np.random.randint(0,100)
            img[j,i,1]=np.random.randint(0,100)
            img[j,i,2]=np.random.randint(0,100)
    return img

def gasuss_noise(image, mean=0, var=0.0005):
    image = np.array(image/255.0, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    # if out.min() < 0:
    #     low_clip = -1.
    # else:
    #     low_clip = 0.
    out = np.clip(out, 0.0, 1.0)
    out = np.uint8(out*255)
    #cv.imshow("gasuss", out)
    return out

def gaussian_blur(img,sigma):
    img_dst = cv2.GaussianBlur(img, (5,5), sigma)
    return img_dst
img= cv2.imread('1.jpg')
img_dst=gasuss_noise(img, mean=0, var=0.005)
cv2.imshow("image_raw",img)
cv2.imshow("image_dst",img_dst )
cv2.waitKey(0)




