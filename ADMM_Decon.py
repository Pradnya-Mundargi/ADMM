#!/usr/bin/env python
# coding: utf-8


import numpy as np
import cv2
import matplotlib.pyplot as plt
import bm3d

# Kernal 1 and Kernal 2
kernal_1= cv2.getGaussianKernel(25,1)
kernal_1= kernal_1* kernal_1.T

kernal_2= cv2.getGaussianKernel(25,3)
kernal_2= kernal_2* kernal_2.T


#Import test Image

import  os
dire=os.getcwd()
img_dir=os.path.join( dire,"house_small.png")
img= cv2.imread(img_dir,0)
plt.imshow(img,'gray')
#img.shape



size = (img.shape[0] - kernal_1.shape[0], img.shape[1] - kernal_1.shape[1]) 
kernal_1= np.pad(kernal_1, (((size[0]+1)//2, size[0]//2), ((size[1]+1)//2, size[1]//2)),  mode='constant')
kernal_2= np.pad(kernal_2, (((size[0]+1)//2, size[0]//2), ((size[1]+1)//2, size[1]//2)),  mode='constant')

noise= np.random.normal(0,1, img.shape)
img_1= (cv2.filter2D(img, -1, kernal_1)) + noise
img_2= (cv2.filter2D(img, -1, kernal_2)) + noise

kernal_1 = np.fft.ifftshift(kernal_1)
kernal_2 = np.fft.ifftshift(kernal_2)


#Fourier transform of images and kernals
dft_kernal_1= np.fft.fft2(np.float32(kernal_1))
dft_kernal_2= np.fft.fft2(np.float32(kernal_2))

dft_1 = np.fft.fft2(np.float32(img_1))
dft_2 = np.fft.fft2(np.float32(img_2))


#Variables
rho=1
x_tilda= np.zeros(img.shape) 
v= np.zeros(img.shape)
u= np.zeros(img.shape)
s_dev= 14



for i in range(100):
    x_tilda= v-u
    x= np.fft.ifft2(((np.conj(dft_kernal_1)*(dft_1)) + (rho * np.fft.fft2(x_tilda))) / (((dft_kernal_1 **2))+rho))
    x=np.real(x)
    v_tilda = x+u
    v= bm3d.bm3d(v_tilda, s_dev)
    u= u+ (x-v)





#Variables4
rho_2=3
x_tilda_2= np.zeros(img.shape) 
v_2= np.zeros(img.shape)
u_2= np.zeros(img.shape)
s_dev_2= 12



# 100 interations for deconvolution using kernel_2
for i in range(100):
    x_tilda_2= v_2-u_2
    x_2= np.fft.ifft2(((np.conj(dft_kernal_2)*(dft_2)) + (rho_2 * np.fft.fft2(x_tilda_2))) / (((dft_kernal_2 **2))+rho_2))
    x_2=np.real(x_2)
    v_tilda_2 = x_2+u_2
    v_2= bm3d.bm3d(v_tilda_2, s_dev_2)
    u_2= u_2+ (x_2-v_2)


#plt.imshow(x_2,'gray')



#plt.imshow(img_2,'gray')


plt.subplot(1, 2, 1)
plt.imshow(x, 'gray')
plt.subplot(1, 2, 2)
plt.imshow(x_2, 'gray')
plt.show()

import math
def PSNR(original, decon):
    mse = np.mean((original - decon) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / np.sqrt(mse))
    return psnr



val=PSNR(img,x_2)
print("PSNR1",val)


val_1=PSNR(img,x)
print("PSNR2",val_1)





