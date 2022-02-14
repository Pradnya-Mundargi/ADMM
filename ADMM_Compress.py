#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
import matplotlib.pyplot as plt
import bm3d



#Input image

import  os
dire=os.getcwd()
img_dir=os.path.join( dire,"house_small.png")
img= cv2.imread(img_dir,0)
#img=cv2.resize(img, (64,64))
img_1=img.copy()
img=img.ravel()
img= cv2.resize(img,(1,img.shape[0]))
plt.imshow(img_1,'gray')
img.shape



#Measurement Matrix
c_1= np.random.normal(0, np.sqrt(1/4096), (4096,128**2))
#c_1.shape
c_2= np.random.normal(0,np.sqrt(1/8192), (8192,128**2))
c_2.shape




y_1= np.dot(c_1,img)
y_2= np.dot(c_2,img)





rho= 1
x_tilda= np.zeros(img.shape) 
v= np.zeros(img.shape)
u= np.zeros(img.shape)
s_dev= 35

I= np.identity(c_1.shape[1])



inv= np.linalg.inv(np.dot(c_1.T,c_1) + (rho*I))
#print(s_dev)

for i in range (100):
    x_tilda= v-u
    x = np.matmul((inv),((np.matmul(np.transpose(c_1),y_1))+(rho*x_tilda))) #4096x1
    v_tilda = x+u #both 4096x1
    v_tilda.resize(128,128) #64x64
    v= bm3d.bm3d(v_tilda, s_dev)
    v= v.ravel() 
    v.resize(v.shape[0],1)
    u= u+ (x-v)





#rho= 0.8
rho= 1
x_tilda= np.zeros(img.shape) 
v= np.zeros(img.shape)
u= np.zeros(img.shape)
#s_dev= 12
s_dev= 35


x=np.resize(x,(128,128))



I= np.identity(c_2.shape[1])
inv= np.linalg.inv(np.dot(c_2.T,c_2) + (rho*I))


for i in range (100):
    x_tilda= v-u
    x2 = np.matmul((inv),((np.matmul(np.transpose(c_2),y_2))+(rho*x_tilda))) #4096x1
    v_tilda = x2+u #both 4096x1
    v_tilda.resize(128,128) #64x64
    v= bm3d.bm3d(v_tilda, s_dev)
    v= v.ravel() 
    v.resize(v.shape[0],1)
    u= u+ (x2-v)



x2=np.resize(x2,(128,128))



import math
def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0): 
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / np.sqrt(mse))
    return psnr




plt.subplot(1, 2, 1)
plt.imshow(x, 'gray')
plt.subplot(1, 2, 2)
plt.imshow(x2, 'gray')
plt.show()




val=PSNR(img_1,x2)
print("PSNR2",val)


val_1=PSNR(img_1,x)
print("PSNR1",val_1)





