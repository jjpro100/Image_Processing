import numpy as np

def gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    #print(shape)
    m,n = [(ss-1.)/2. for ss in shape]
    #print(m,"\n")
    #print(n)
    y,x = np.ogrid[-m:m+1,-n:n+1]
    #print(y)
    #print(x)
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    #print(h)
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    #print(h)
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def zcr(image, block=(3,3),threshold = 10):
    #print(block)
    image_z = np.zeros(image.shape)
    #print(image_z)
    for i in range(0, image.shape[0]-(block[0]-1)):
        for j in range(0, image.shape[1]-(block[1]-1)):
            image_z[i + 1, j + 1] = evaluate_block_image4(image[i:i+block[0],j:j+block[1]], threshold)

    image_z = image_z/np.max(image_z)
    image_z[image_z > threshold] = 255
    image_z[image_z <= threshold] = 0

    return image_z

def evaluate_block_image2(image, threshold):
    smim = np.array(image)
    zc = 0
    if smim[1, 1] > 0:
        if (smim[1, 2] >= 0 and smim[1, 0] < 0) or  (smim[1, 2] < 0 and smim[1, 0] >= 0):
            zc = smim[1, 2]

        elif (smim[2, 1] >= 0 and smim[0, 1] < 0) or (smim[2, 1] < 0 and smim[0, 1] >= 0):
            zc = smim[1, 2]

        elif (smim[2, 2] >= 0 and smim[0, 0] < 0) or (smim[2, 2] < 0 and smim[0, 0] >= 0):
            zc = smim[1, 2]

        elif (smim[0, 2] >= 0 and smim[2, 0] < 0) or (smim[0, 2] < 0 and smim[2, 1] >= 0):
            zc = smim[1, 2]

    return zc

def evaluate_block_image3(image, threshold):
    smim = np.array(image)
    zc = 0
    if smim[1, 1] > 0:
        if (smim[1, 2] >= 0 and smim[1, 0] < 0) or  (smim[1, 2] < 0 and smim[1, 0] >= 0):
            zc = max(smim[1, 2], smim[1, 0])

        elif (smim[2, 1] >= 0 and smim[0, 1] < 0) or (smim[2, 1] < 0 and smim[0, 1] >= 0):
            zc = max(smim[2, 1], smim[0, 1])

        elif (smim[2, 2] >= 0 and smim[0, 0] < 0) or (smim[2, 2] < 0 and smim[0, 0] >= 0):
            zc = max(smim[2, 2], smim[0, 0])

        elif (smim[0, 2] >= 0 and smim[2, 0] < 0) or (smim[0, 2] < 0 and smim[2, 1] >= 0):
            zc = max(smim[0, 2], smim[2, 0])

    return zc

def evaluate_block_image4(image, threshold):
    smim = np.array(image)
    zc = 0
    if smim[1, 1] > 0:
        if (smim[1, 2] >= 0 and smim[1, 0] < 0) or  (smim[1, 2] < 0 and smim[1, 0] >= 0):
            zc = abs(smim[1, 2]) + abs(smim[1, 0])

        elif (smim[2, 1] >= 0 and smim[0, 1] < 0) or (smim[2, 1] < 0 and smim[0, 1] >= 0):
            zc = abs(smim[2, 1]) + abs(smim[0, 1])

        elif (smim[2, 2] >= 0 and smim[0, 0] < 0) or (smim[2, 2] < 0 and smim[0, 0] >= 0):
            zc = abs(smim[2, 2]) + abs(smim[0, 0])

        elif (smim[0, 2] >= 0 and smim[2, 0] < 0) or (smim[0, 2] < 0 and smim[2, 1] >= 0):
            zc = abs(smim[0, 2]) + abs(smim[2, 0])

    return zc


def find_nearest(array, value, get_index=False):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if get_index:
        return idx
    else:
        return array[idx]


class HoughTransform():
    def __init__(self, rho, theta):
        self.rho = np.array(rho)
        self.theta = np.array(theta)
        self.img_prt = np.zeros((len(rho), len(theta)))

        self.prt_available = False

    def run(self, image):
        self.image = np.copy(image)
        self.line = 0
        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):
                if image[i, j] > 1:
                    self.line += 1
                    for angle in self.theta:
                        p = int(round(i * np.cos(np.pi * angle / 180) + j * np.sin(np.pi * angle / 180)))
                        self.img_prt[find_nearest(self.rho, p, get_index=True), self.theta == angle] += 1

                # print("printing curve in Hough image...{}".format(self.line))

        self.prt_available = True
        return np.copy(self.img_prt)

    def crop_by_region(self, image, region):
        self.image = np.copy(image)
        aux_image = np.zeros(self.image.shape)
        if len(region) == 4:
            aux_image[region[0]:region[2], region[1]:region[3]] = 1
            self.image = self.image * aux_image
        else:
            return False

        return np.copy(self.image)

    def getting_lines_by_angle(self, theta):
        if not self.prt_available:
            return False

        i_theta = self.theta == theta
        max_item = np.where(self.img_prt == np.max(self.img_prt[:,i_theta]))

        rho = self.rho[max_item[0]]

        aux_image = np.zeros(self.image.shape)
        if not np.isclose(theta, 0.0):
            m = np.cos(np.pi * theta / 180)/np.sin(np.pi * theta / 180)
            b = rho[0]/np.sin(np.pi * theta / 180)
            for i in np.linspace(0, self.image.shape[0]-1, self.image.shape[0]*1000):
                j = int(round(b - m*i))
                if aux_image.shape[1]>j and j > 0:
                    aux_image[int(round(i)), j] = 255
        else:
            aux_image[rho, :] = 255

        return np.copy(aux_image)

    def getting_max_parameter(self, exclude=0):
        if not self.prt_available:
            return False

        if not exclude==0:
            max_item = np.where(self.img_prt == np.max(self.img_prt[self.img_prt < exclude]))
            max_value = np.max(self.img_prt[self.img_prt < exclude])
        else:
            max_item = np.where(self.img_prt == np.max(self.img_prt))
            max_value = np.max(self.img_prt)

        rho = self.rho[max_item[0]]
        theta = self.theta[max_item[1]]
        return rho, theta, max_value

    def getting_image_by_angle_and_radio(self, theta, rho):

        aux_image = np.zeros(self.image.shape)
        if not np.isclose(theta, 0.0):
            m = np.cos(np.pi * theta / 180)/np.sin(np.pi * theta / 180)
            b = rho/np.sin(np.pi * theta / 180)
            for i in np.linspace(0, self.image.shape[0]-1, self.image.shape[0]*100):
                j = int(round(b - m*i))
                if aux_image.shape[1]>j and j > 0:
                    aux_image[int(round(i)), j] = 255
        else:
            aux_image[rho, :] = 255

        return aux_image

    def get_radio_by_angle(self, theta):
        if not self.prt_available:
            return False

        i_theta = self.theta == theta
        max_item = np.where(self.img_prt == np.max(self.img_prt[:,i_theta]))

        rho = self.rho[max_item[0]]

        return rho