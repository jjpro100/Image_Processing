import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import math

from lib import gauss2D, zcr, HoughTransform

img = cv2.imread("C:/Users/user/Documents/Image Processing/DIP_project2/dew on roses (color).tif", -1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
red = img[:,:,0]
green = img[:,:,1]
blue = img[:,:,2]

img_G_equ = cv2.cvtColor(cv2.imread("C:/Users/user/Documents/Image Processing/DIP_project2/dew on roses (color).tif", -1), cv2.COLOR_BGR2RGB)
equ = cv2.equalizeHist(img_G_equ[:,:,2]) 
img_G_equ[:,:,2] = equ


#img_air = cv2.imread("airplane in the sky.tif", -1)
img_air = cv2.imread("C:/Users/user/Documents/Image Processing/DIP_project2/house.jpg", -1)

gauss = gauss2D((5,5), sigma=4)*-1
#print(gauss)
lap = np.ones((3, 3), np.float32)
lap[1, 1] = -8

img_gaussian = cv2.filter2D(img_air, cv2.CV_64F, gauss)
img_LoG = cv2.filter2D(img_gaussian, cv2.CV_64F, lap)

img_zcr = zcr(img_LoG, threshold=0.04)
img_zcr0 = zcr(img_LoG, threshold=0)
'''
zero_crossing = np.zeros_like(img_LoG)

log = img_LoG
kern_size = 3
	# computing zero crossing
for i in range(log.shape[0]-(kern_size-1)):
    for j in range(log.shape[1]-(kern_size-1)):
        if log[i][j] == 0:
            if (log[i][j-1] < 0 and log[i][j+1] > 0) or (log[i][j-1] < 0 and log[i][j+1] < 0) or (log[i-1][j] < 0 and log[i+1][j] > 0) or (log[i-1][j] > 0 and log[i+1][j] < 0):
                zero_crossing[i][j] = 255
        if log[i][j] < 0:
            if (log[i][j-1] > 0) or (log[i][j+1] > 0) or (log[i-1][j] > 0) or (log[i+1][j] > 0):
                zero_crossing[i][j] = 255
'''
'''
img = cv2.imread("C:/Users/user/Documents/Image Processing/DIP_project2/airplane in the sky.tif", -1)
edges = cv2.Canny(img,50,150,apertureSize = 3)

lines = cv2.HoughLines(edges,5,2*np.pi/180,1)
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imwrite('houghlines3.jpg',img)
'''

#############
'''
houg = HoughTransform(rho=range(0, 721, 5), theta=range(0, 91, 2))

img_zcr_w = []
img_zcr_w.append(houg.crop_by_region(img_zcr, [315, 400, 355, 483]))
img_zcr_w.append(houg.crop_by_region(img_zcr, [315, 390, 330, 483]))
img_zcr_w.append(houg.crop_by_region(img_zcr, [315, 400, 355, 410]))
img_zcr_w.append(houg.crop_by_region(img_zcr, [315, 473, 355, 483]))

plt.figure(9)
plt.imshow(img_zcr,'gray')

img_hough_parameters = houg.run(img_zcr_w[1])
max_item = houg.getting_max_parameter()
image_lines = houg.getting_lines_by_angle(max_item[1][0])
image_lines_2 = houg.getting_image_by_angle_and_radio(90, 408)
image_lines_3 = houg.getting_image_by_angle_and_radio(9, 409)
image_lines_4 = houg.getting_image_by_angle_and_radio(90, 480)

print(max_item)
plt.figure(10)
plt.imshow(img_hough_parameters)
'''

'''
plt.figure(11)
plt.imshow(image_lines+img+image_lines_2+image_lines_3+image_lines_4)


new_image = np.zeros(img_zcr.shape)
new_image[(image_lines+img_zcr_w[0]+image_lines_2+image_lines_3+image_lines_4) > 255] = 255


plt.figure(12)
plt.imshow(new_image + img)
'''
#############

plt.subplot(3,4,1), plt.imshow(img)
plt.title('1. Original'), plt.xticks([]), plt.yticks([])
cv2.imwrite('data_1/1.tif',cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(3,4,2), plt.imshow(red)
plt.title('2. Red'), plt.xticks([]), plt.yticks([])
cv2.imwrite('data_1/2.tif',red)
plt.subplot(3,4,3), plt.imshow(green)
plt.title('3. Green'), plt.xticks([]), plt.yticks([])
cv2.imwrite('data_1/3.tif',green)
plt.subplot(3,4,4), plt.imshow(blue)
plt.title('4. Blue'), plt.xticks([]), plt.yticks([])
cv2.imwrite('data_1/4.tif',blue)
plt.subplot(3,4,5), plt.imshow(img_G_equ)
plt.title('5. Green_equ'), plt.xticks([]), plt.yticks([])
cv2.imwrite('data_1/5.tif', cv2.cvtColor(img_G_equ, cv2.COLOR_BGR2RGB))

plt.subplot(3,4,6), plt.imshow(img_air, 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
cv2.imwrite('data_2/1.tif', img_air)
plt.subplot(3,4,7), plt.imshow(img_gaussian, 'gray')
plt.title('Gaussian'), plt.xticks([]), plt.yticks([])
cv2.imwrite('data_2/2.png', img_gaussian)
plt.subplot(3,4,8), plt.imshow(img_LoG, 'gray')
plt.title('Lap'), plt.xticks([]), plt.yticks([])
cv2.imwrite('data_2/3.tif', img_LoG)
plt.subplot(3,4,9), plt.imshow(img_zcr, 'gray')
plt.title('Zero_Crossing_4%'), plt.xticks([]), plt.yticks([])
cv2.imwrite('data_2/4.tif', img_zcr)
plt.subplot(3,4,10), plt.imshow(img_zcr0, 'gray')
plt.title('Zero_Crossing_0%'), plt.xticks([]), plt.yticks([])
cv2.imwrite('data_2/5.tif', img_zcr0)



plt.figure(2)
plt.imshow(img_gaussian, 'gray')
plt.title('Gaussian'), plt.xticks([]), plt.yticks([])
plt.figure(3)
plt.imshow(img_LoG, 'gray')
plt.title('img_LoG'), plt.xticks([]), plt.yticks([])
plt.figure(4)
plt.imshow(img_zcr, 'gray')
plt.title('Zero_Crossing_4%'), plt.xticks([]), plt.yticks([])
plt.figure(5)
plt.imshow(img_zcr0, 'gray')
plt.title('Zero_Crossing_0%'), plt.xticks([]), plt.yticks([])



plt.figure(6)
plt.imshow(red)
plt.title('Red'), plt.xticks([]), plt.yticks([])
plt.figure(7)
plt.imshow(green)
plt.title('Green'), plt.xticks([]), plt.yticks([])
plt.figure(8)
plt.imshow(blue)
plt.title('Blue'), plt.xticks([]), plt.yticks([])

plt.show()