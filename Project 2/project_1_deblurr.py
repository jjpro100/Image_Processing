import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import math
import cmath


#400 0.003 - 0.001
img = cv2.imread("dew on roses (blurred).tif", -1)
m , n = img.shape
kernel = [[1 for x in range(m)] for y in range(n)] 
a = 0.01
b = a / math.tan(45*math.pi/180)
for u in range(m):
    for v in range(n):
        t = u - m/2
        s = v - n/2
        term = math.pi*(t*a+s*b)
        if term == 0:
            kernel[u][v] = 1
        else:
            kernel[u][v] = (1/term) * math.sin(term) * math.e**(-1j*term) #*8000


#print("hola")

f1 = np.fft.fft2(img)
fshift1 = np.fft.fftshift(f1)
#deb = fshift1 / kernel
magnitude_spectrum1 = 20*np.log(np.abs(fshift1) + 1)
deb = fshift1 / kernel

'''
h = np.fft.fft2(kernel)
hfshift1 = np.fft.fftshift(h)
#deb = fshift1 / kernel
hmagnitude_spectrum1 = 20*np.log(np.abs(hfshift1) + 1)
'''

crow,ccol = m/2 , n/2
for r in range(m):
    for c in range(n):
        if math.sqrt((r-crow)**2+(c-ccol)**2) >45 :
            deb[r][c] = 0


hmagnitude_spectrum1 = 20*np.log(np.abs(kernel) + 1)

f_ishift = np.fft.ifftshift(deb)
img_back = np.fft.ifft2(f_ishift)
cured = np.abs(img_back)

o = np.array(np.abs(fshift1))
p = np.array(np.abs(kernel))

'''
y = cured.ravel()
##################################

n = len(y) # length of the signal
k = np.arange(n)
T = 1
frq = k/T # two sides frequency range
frq = frq[range(math.floor(n/2))] # one side frequency range

Y = np.fft.fft(y)/n # fft computing and normalization
Y = Y[range(math.floor(n/2))]
'''

fig, ax = plt2.subplots(2, 1)
ax[0].plot(o[:,200])
ax[0].plot(p[:,200],'r')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')
ax[1].plot(o[387,:])
ax[1].plot(p[387,:],'r')
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('|Y(freq)|')
plt2.show()
#plot_url = py.plot_mpl(fig, filename='mpl-basic-fft')
##################################


plt.subplot(3,2,1), plt.imshow(img,'gray')
plt.title('1. Original'), plt.xticks([]), plt.yticks([])
cv2.imwrite('data_4/1.tif',img)

plt.subplot(3,2,2), plt.imshow(magnitude_spectrum1,'gray')
plt.title('2. DFT'), plt.xticks([]), plt.yticks([])
cv2.imwrite('data_4/2.tif',magnitude_spectrum1)

plt.subplot(3,2,3), plt.imshow(cured,'gray')
plt.title('3. Deblurred'), plt.xticks([]), plt.yticks([])
cv2.imwrite('data_4/3.tif',cured)

plt.subplot(3,2,4), plt.imshow(hmagnitude_spectrum1,'gray')
plt.title('4. H function'), plt.xticks([]), plt.yticks([])
cv2.imwrite('data_4/4.tif',hmagnitude_spectrum1)

#plt.subplot(3,2,4), plt2.plot(frq,abs(Y),'r')
#plt.title('3. 1D fft'), plt.xticks([]), plt.yticks([])
#cv2.imwrite('data_4/4.tif',cured)

cv2.waitKey(0)
plt.show()
