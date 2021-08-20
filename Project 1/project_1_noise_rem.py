import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import math


#Original image
img_0 = cv2.imread("C:/Users/user/Documents/Image Processing/Project 1/dew on roses (noisy).tif", 0)
#k = np.array(img_0)
#k = k.ravel()
#plt.hist(k,256,[0,256]); plt.show()

#hist = cv2.calcHist(k,[0],None,[256],[0,256])
#hist,bins = np.histogram(k,256,[0,256])
#plt.plot(hist)
#plt.hist(k) 
#plt.hist(k,256,[0,256])
#plt.show()

f1 = np.fft.fft2(img_0)
fshift1 = np.fft.fftshift(f1)
magnitude_spectrum1 = 20*np.log(np.abs(fshift1) + 1)
plt2.imshow(magnitude_spectrum1,'gray')
plt2.show()

#Remove noise
img_original = cv2.GaussianBlur(img_0,(5,5),0,0)
cv2.imwrite('data_3/0.tif',img_original)
img_original = cv2.GaussianBlur(img_0,(5,5),0.4,0.4)
cv2.imwrite('data_3/04.tif',img_original)
img_original = cv2.GaussianBlur(img_0,(5,5),0.2,0.2)
cv2.imwrite('data_3/02.tif',img_original)
img_original = cv2.GaussianBlur(img_0,(5,5),0.3,0.3)
cv2.imwrite('data_3/03.tif',img_original)

median = cv2.medianBlur(img_0,5)
# cv2.imwrite('data_3/med5.tif',median)
# k = np.array(median)
# k = k.ravel()
# plt.hist(k,256,[0,256])
# plt.show()
median = cv2.medianBlur(img_0,3)
cv2.imwrite('data_3/med3.tif',median)
median = cv2.medianBlur(img_0,9)
cv2.imwrite('data_3/med10.tif',median)
median = cv2.medianBlur(img_0,7)
cv2.imwrite('data_3/med7.tif',median)

#Mean zero and standard deviation between 0.2 and 0.4 (Gaussian noise)
#Convolve the source image with the specified Gaussian kernel