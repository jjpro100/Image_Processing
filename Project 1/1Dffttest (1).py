# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
import cmath
import math
#squareimpulse = np.array([0,0,0,0,0,1,1,1,1,1,0,0,0,0,0])

img = cv2.imread("dew on roses (noisy).tif", -1)
img1 = cv2.imread("dew on roses (blurred).tif", -1)
#img = cv2.imread('book.tif',-1)
#img = cv2.imread("dew on roses (blurred).tif", -1)


f = np.fft.fft2(img)
fshift2 = np.fft.fftshift(f)
fshift_mag = 20*np.log(np.abs(fshift2))
f1 = np.fft.fft2(img1)
fshift1 = np.fft.fftshift(f1)
fshift_mag1 = 20*np.log(np.abs(fshift1))


x , y = img.shape
#a = 0.0020101
#b = -0.0018101
a = 0.005
b = 0.005
T = 1
H = [[1 for x in range(x)] for y in range(y)]
for u in range(0,x):
    for v in range(0,y):
        z = u - x/2
        k = v - y/2
        C = math.pi * (z*a + k*b)
        if C == 0:
            H[u][v] = 1
        else:
            H[u][v] = (T/C) * math.sin(C) * math.e**(-1j*C)
            
                
#H = np.fft.fftshift(H)
fft_Ans = H * fshift2
fft_Ans_magnitude = 20*np.log(np.abs(fft_Ans))


f_fft_Ans = np.fft.ifftshift(fft_Ans)
filtered_Ans = np.fft.ifft2(f_fft_Ans)
filtered_Ans = np.absolute(filtered_Ans)

temp = fshift1
rows, cols = img1.shape
crow,ccol = rows/2 , cols/2
aftdeg = fshift1 / H

for r in range(rows):
    for c in range(cols):
        if math.sqrt((r-crow)**2+(c-ccol)**2) > 70 :
            aftdeg[r][c] = 0

f_aftdeg = np.fft.ifftshift(aftdeg)
filtered_aft = np.fft.ifft2(f_aftdeg)
filtered_aft = np.absolute(filtered_aft)


plt.figure()
plt.subplot(121)
plt.imshow(img,'gray')
plt.title('Input noisy')
plt.xticks([]), plt.yticks([])


plt.subplot(122)
plt.imshow(filtered_Ans,'gray')
plt.title('After degradation')
plt.xticks([]), plt.yticks([])
plt.show()
plt.figure()
plt.subplot(121)
plt.imshow(img1,'gray')
plt.title('Input blurred')
plt.xticks([]), plt.yticks([])

plt.subplot(122)
plt.imshow(filtered_aft,'gray')
plt.title('After Inverse')
plt.xticks([]), plt.yticks([])
plt.show()
#plt.subplot(325)
#plt.imshow(fshift_mag,'gray')
#plt.title('Input noisy fft')
#plt.xticks([]), plt.yticks([])
#
#plt.subplot(326)
#plt.imshow(fshift_mag1,'gray')
#plt.title('Input blurred fft')
#plt.xticks([]), plt.yticks([])
#
#
#plt.show()
#
#plt.figure()
#
#plt.imshow(fft_Ans_magnitude,'gray')
#plt.title('fft of inverse')
#plt.xticks([]), plt.yticks([])
#
#
#plt.show()
#cv2.imwrite('filtered.tif',filtered_aft)