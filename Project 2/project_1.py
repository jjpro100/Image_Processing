import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import math

def ToU8(image):
    return 255 * (image - image.min()) / (image.max() - image.min())

##  Problem 1  ##

kernel = np.ones((3, 3), np.float32)
kernel[1, 1] = -8
'''
kernel[1, 1] = -4
kernel[0,0] = 0
kernel[0,2] = 0
kernel[2,0] = 0
kernel[2,2] = 0
'''
kernel*=-1

#Original image
img_0 = cv2.imread("dew on roses (noisy).tif", -1)
#Remove noise
img_original = img_0
#img_original = cv2.GaussianBlur(img_0,(3,3),0)

###################################################
#Laplacian
#img_lap = cv2.filter2D(img_original, cv2.CV_8U, kernel)
img_lap = ToU8(cv2.filter2D(img_original, cv2.CV_64F, kernel))
#Sharpened image
#img_sharpened = ToU8(img_original) + img_lap
img_sharpened = img_original - img_lap
#Sobel Filter
img_sobelx = cv2.Sobel(img_original,cv2.CV_64F,1,0,ksize=3)
img_sobely = cv2.Sobel(img_original,cv2.CV_64F,0,1,ksize=3)
img_sobel = abs(img_sobely) + abs(img_sobelx)
#img_sobel = ToU8(img_sobel)
#Averaging
kernel = np.ones((5,5), np.float32)/25
avg_img = ToU8(cv2.filter2D(img_sobel, -1, kernel))
#Product of Avg_img and Sharpened
Prod_avg_and_sharp = ToU8(img_sharpened*avg_img)
#Sum of original and product
Sum_of_ori_and_prod_avg_and_sharp = ToU8(Prod_avg_and_sharp+img_original)
#Power trnsformation
img_power_trans = ToU8(cv2.pow(Sum_of_ori_and_prod_avg_and_sharp, 0.9))
###################################################

plt.subplot(3,4,1), plt.imshow(img_0,'gray')
plt.title('1. Original'), plt.xticks([]), plt.yticks([])
cv2.imwrite('data_1/1.tif',img_0)
#plt.subplot(3,4,2), plt.imshow(img_original,'gray')
#plt.title('2. Original \n(Noise Removed)'), plt.xticks([]), plt.yticks([])
cv2.imwrite('data_1/2.tif',img_original)
plt.subplot(3,4,3), plt.imshow(img_lap,'gray')
plt.title('3. Laplacian'), plt.xticks([]), plt.yticks([])
cv2.imwrite('data_1/3.tif',img_lap)
plt.subplot(3,4,4), plt.imshow(img_sharpened,'gray')
plt.title('4. Sharpened:\nOri + Lap'), plt.xticks([]), plt.yticks([])
cv2.imwrite('data_1/4.tif',ToU8(img_sharpened))
plt.subplot(3,4,5), plt.imshow(img_sobelx,'gray')
plt.title('5. Sobel_x'), plt.xticks([]), plt.yticks([])
cv2.imwrite('data_1/5.tif',img_sobelx)
plt.subplot(3,4,6), plt.imshow(img_sobely,'gray')
plt.title('6. Sobel_y'), plt.xticks([]), plt.yticks([])
cv2.imwrite('data_1/6.tif',img_sobely)
plt.subplot(3,4,7), plt.imshow(img_sobel,'gray')
plt.title('7. Sobel_sum'), plt.xticks([]), plt.yticks([])
cv2.imwrite('data_1/7.tif',img_sobel)
plt.subplot(3,4,8), plt.imshow(avg_img,'gray')
plt.title('8. Avg_img'), plt.xticks([]), plt.yticks([])
cv2.imwrite('data_1/8.tif',avg_img)
plt.subplot(3,4,9), plt.imshow(Prod_avg_and_sharp,'gray')
plt.title('9. Sharp x Avg'), plt.xticks([]), plt.yticks([])
cv2.imwrite('data_1/9.tif',Prod_avg_and_sharp)
plt.subplot(3,4,10), plt.imshow(Sum_of_ori_and_prod_avg_and_sharp,'gray')
plt.title('10. Ori + (Sharp x Avg)'), plt.xticks([]), plt.yticks([])
cv2.imwrite('data_1/10.tif',Sum_of_ori_and_prod_avg_and_sharp)
plt.subplot(3,4,11), plt.imshow(img_power_trans,'gray')
plt.title('11. pow(Ori + (Sharp x Avg))'), plt.xticks([]), plt.yticks([])
cv2.imwrite('data_1/11.tif',img_power_trans)

plt.show()

##  Problem 2  ##

#Original image
img_1 = cv2.imread("dew on roses (noisy).tif", 0)
img_2 = cv2.imread("tulips irises.tif", 0)
#Remove noise
img_1 = cv2.GaussianBlur(img_0,(3,3),0)
#DFT
f1 = np.fft.fft2(img_1)
fshift1 = np.fft.fftshift(f1)
magnitude_spectrum1 = 20*np.log(np.abs(fshift1) + 1)

f2 = np.fft.fft2(img_2)
fshift2 = np.fft.fftshift(f2)
magnitude_spectrum2 = 20*np.log(np.abs(fshift2) + 1)

#Filtering and performing idft

################################################
#HP filter img_1
temp = fshift1
rows, cols = img_1.shape
crow,ccol = rows/2 , cols/2
for r in range(rows):
    for c in range(cols):
        if math.sqrt((r-crow)**2+(c-ccol)**2) < 30:
            temp[c][r] = 0

img_1_freq_dom_after_filt_H = 20*np.log(np.abs(temp)+1)

f_ishift = np.fft.ifftshift(temp)
img_back = np.fft.ifft2(f_ishift)
img_1_H = np.abs(img_back)

#LP filter img_1
temp = np.fft.fftshift(f1)
rows, cols = img_1.shape
crow,ccol = rows/2 , cols/2
for r in range(rows):
    for c in range(cols):
        if math.sqrt((r-crow)**2+(c-ccol)**2) > 30:
            temp[c][r] = 0

img_1_freq_dom_after_filt_L =  20*np.log(np.abs(temp)+1)

f_ishift = np.fft.ifftshift(temp)
img_back = np.fft.ifft2(f_ishift)
img_1_L = np.abs(img_back)

################################################

# img_2
#HP filter img_2
temp = fshift2
rows, cols = img_2.shape
crow,ccol = rows/2 , cols/2
for r in range(rows):
    for c in range(cols):
        if math.sqrt((r-crow)**2+(c-ccol)**2) < 30:
            temp[c][r] = 0

img_2_freq_dom_after_filt_H =  20*np.log(np.abs(temp)+1)

f_ishift = np.fft.ifftshift(temp)
img_back = np.fft.ifft2(f_ishift)
img_2_H = np.abs(img_back)

#LP filter img_2

temp = np.fft.fftshift(f2)
rows, cols = img_2.shape
crow,ccol = rows/2 , cols/2
for r in range(rows):
    for c in range(cols):
        if math.sqrt((r-crow)**2+(c-ccol)**2) > 30:
            temp[c][r] = 0

img_2_freq_dom_after_filt_L =  20*np.log(np.abs(temp)+1)

f_ishift = np.fft.ifftshift(temp)
img_back = np.fft.ifft2(f_ishift)
img_2_L = np.abs(img_back)

################################################

plt2.subplot(3,4,1), plt.imshow(magnitude_spectrum1,'gray')
plt2.title('1. DFT dew dB'), plt.xticks([]), plt.yticks([])
cv2.imwrite('data_2/1.tif',magnitude_spectrum1)
plt2.subplot(3,4,2), plt.imshow(magnitude_spectrum2,'gray')
plt2.title('2. DFT tulips dB'), plt.xticks([]), plt.yticks([])
cv2.imwrite('data_2/2.tif',magnitude_spectrum2)
plt.subplot(3,4,3), plt.imshow(img_1_freq_dom_after_filt_H,'gray')
plt.title('3. img_1_H_freq'), plt.xticks([]), plt.yticks([])
cv2.imwrite('data_2/3.tif',img_1_freq_dom_after_filt_H)
plt.subplot(3,4,4), plt.imshow(img_1_freq_dom_after_filt_L,'gray')
plt.title('4. img_1_L_freq'), plt.xticks([]), plt.yticks([])
cv2.imwrite('data_2/4.tif',img_1_freq_dom_after_filt_L)
plt.subplot(3,4,5), plt.imshow(img_2_freq_dom_after_filt_H,'gray')
plt.title('5. img_2_H_freq'), plt.xticks([]), plt.yticks([])
cv2.imwrite('data_2/5.tif',img_2_freq_dom_after_filt_H)
plt.subplot(3,4,6), plt.imshow(img_2_freq_dom_after_filt_L,'gray')
plt.title('6. img_2_L_freq'), plt.xticks([]), plt.yticks([])
cv2.imwrite('data_2/6.tif',img_2_freq_dom_after_filt_L)
plt.subplot(3,4,7), plt.imshow(img_1_H,'gray')
plt.title('7. img_1_H'), plt.xticks([]), plt.yticks([])
cv2.imwrite('data_2/7.tif',img_1_H)
plt.subplot(3,4,8), plt.imshow(img_1_L,'gray')
plt.title('8. img_1_L'), plt.xticks([]), plt.yticks([])
cv2.imwrite('data_2/8.tif',img_1_L)
plt.subplot(3,4,9), plt.imshow(img_2_H,'gray')
plt.title('9. img_2_H'), plt.xticks([]), plt.yticks([])
cv2.imwrite('data_2/9.tif',img_2_H)
plt.subplot(3,4,10), plt.imshow(img_2_L,'gray')
plt.title('10. img_2_L'), plt.xticks([]), plt.yticks([])
cv2.imwrite('data_2/10.tif',img_2_L)


plt2.show()