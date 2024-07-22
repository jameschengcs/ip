# VGI - Vision, Graphics, and Imaging 
# Common module
# (c) 2022, Chang-Chieh Cheng, jameschengcs@nycu.edu.tw

import math
import numpy as np 
import cv2 
import copy
import torch
import torch.nn.functional as F 
import scipy.stats as stats
import matplotlib.pyplot as plt 

from os import listdir
from os.path import isfile, join
from PIL import Image 
from scipy import ndimage
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse

__all__ = ()
__all__ += ('currentTime', 'showImageTable', 
            'resample', 'polyMatrix2d', 'lintrans2d', 'polyfit2d',
            'boundary', 'batchIdx', 'evaluateImage', 'sobel3d',
            'rpad', 'lpad', 'pround', 'truncatedNormal', 'getFiles', 'normalizeRange', 'normalize',
            'loadImg', 'loadImage', 'loadGrayImage',  'loadVolume',
            'grayToRGB', 'reverseRGB', 'toU8', 'inverseIntensity',
            'saveImage', 'saveImg', 'saveGrayImage', 'saveGrayImg',
            'showImage', 'showImg', 'subimage', 'imagePaste', 'imageMSE','imageBoundary',
            'mostIntensity', 'mostRGB', 'downSample', 'resize',
            'entropy', 'linear2', 'parsePath', 'shrinkImg', 
            'createPyramid', 'createPyramidEx', 'findVLines', 'metric',
            'cartLoc', 'cart2pol', 'pol2cart', 'polarLocation', 
            'sigmoidExp', 'sigmoid', 'dsigmoid',
            'isCellInRect', 'rectOnRect', 'isRectOnRect', 
            'rotatex', 'rotatey', 'rotatez', 'rotate', 
            'rotatexd', 'rotateyd', 'rotatezd', 'rotated',
            'translate', 'scale', 'box', 
            'ImageSetFig', 'unfoldImage', 'unfoldCenterImage',
            'clone', 'toNumpy', 'toNumpyImage', 'toTorchImage', 'showTorchImage'            
            )

def currentTime(format = "%y/%m/%d@%H:%M:%S"):
    now = datetime.now()
    return now.strftime(format)

def resample(x, factor, kind='linear'):
    n = np.ceil(x.size / factor).astype(np.int32)
    f = interp1d(np.linspace(0, 1, x.size), x, kind)
    return f(np.linspace(0, 1, n))

def polyMatrix2d(x, y, deg = 2):
    n = len(x)
    n_factors = deg * 2 + 1
    A_shape = (n, n_factors)
    A = np.ones(A_shape)
    cx = 2
    cy = 1
    xp = np.array(x)
    yp = np.array(y)
    for i in range(deg):
        A[:, cx] = xp
        A[:, cy] = yp
        xp *= x
        yp *= y
        cx += 1
        cy += 1
    return A

def lintrans2d(x, y, w, deg = 2):
    A = polyMatrix2d(x, y, deg)
    z = np.matmul(A, w)
    return z
    
def polyfit2d(x, y, z, deg = 2):
    A = polyMatrix2d(x, y, deg)
    A_shape = A.shape
    n, n_factors = A.shape
    z = z.reshape((n, 1))    
    r = np.linalg.lstsq(A, z, rcond=None)
    return r

# Gicen the data size n and batch size, creating a list to contain the ranges of all batches
def batchIdx(n, batch_size):
    i = 0
    j = batch_size
    idx = []
    while i < n:
        idx += [(i, j)]
        i = j
        j += batch_size
        if j > n:
            j = n
    return idx

# data: (images, height, width, [channels])
def sobel3d(data, normal = True, return_all = False):
    dz = ndimage.sobel(data, axis= 0)
    dy = ndimage.sobel(data, axis= 1)
    dx = ndimage.sobel(data, axis= 2)
    edge = np.sqrt(dx * dx + dy * dy + dz * dz)
    if normal:
        edge = normalize(edge)
    if return_all:
        return edge, dx, dy, dz
    else:
        return edge

def rpad(s, n, c = '0'):
    m = n - len(s)
    if m <= 0:
        return s
    k = len(c)
    u = int(math.ceil(m / k))
    s = (s + c * u)[:n]
    return s

def lpad(s, n, c = '0'):
    m = n - len(s)
    if m <= 0:
        return s
    k = len(c)
    u = int(math.ceil(m / k))
    s = (c * u + s)
    s = s[len(s) - n:]
    return s

def pround(x, k):
    s = str(np.around(x, decimals=k))
    sd, sf = s.split('.')
    sf = rpad(sf, k)
    return sd + '.' + sf

# Random numeber generator from truncated normal distributions
# lower, upper, loc, and scale are 1D ndarrays
# size is a non-negative integer to indicate the number of samples
def truncatedNormal(lower, upper, loc, scale, size):
    n_dim = loc.shape[0]
    A = np.zeros([size, n_dim])
    for i in range(n_dim):
        loc_i = loc[i]
        scale_i = scale[i]
        A_i = stats.truncnorm.rvs((lower[i] - loc_i) / scale_i, (upper[i] - loc_i) / scale_i, loc=loc_i, scale=scale_i, size = size)
        A[:, i] = A_i
    return A

def getFiles(path, sort = True):
    Files = []
    for f in listdir(path):
        f = join(path, f)
        if isfile(f):
            Files.append(f)
    if sort:
        Files.sort()
    return Files

def normalizeRange(A, source_min, source_d, target_min = 0.0, target_d = 1.0): 
    B = (A - source_min) / source_d * target_d + target_min
    return B    

def normalize(A, minimum = 0.0, maximum = 1.0): 
    mini = np.min(A)
    maxi = np.max(A)
    #B = (A - mini) / (maxi - mini) * (maximum - minimum) + minimum
    #return B
    return normalizeRange(A, mini, maxi - mini, minimum, maximum - minimum)

            

def loadImg(path, normalize = True, gray = True):
    print('loadImg() will be deprecated in the future version, please use loadImage()')
    if gray:
        img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    else:
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        
    if normalize:
        grayscale = 255.0
        if(img.dtype == np.uint16):
            grayscale = 65535.0
        imgData = np.asarray(img, dtype = np.float32) / grayscale
        return imgData, img.dtype    
    else:
        return img, img.dtype

def loadImage(path, normalize = True, gray = False):
    if gray:
        img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        img = np.expand_dims(img, axis=2)
    else:
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        
    if normalize:
        grayscale = 255.0
        if(img.dtype == np.uint16):
            grayscale = 65535.0
        imgData = np.asarray(img, dtype = np.float32) / grayscale
        return imgData  
    else:
        return img

def loadGrayImage(path, normalize = True):
    return loadImage(path, normalize, gray = True)      


# a volume has multiple single-channel image files in a directry (path) with the same size and format
# the value of each voxel is in [0, 1]
def loadVolume(path, sort = True):
    Files = getFiles(path, sort)
    #V = []
    imgType = None
    b1st = True
    for f in Files:
        img, imgType = loadImg(f, normalize = True, gray = True)
    #   V.append(img)
        if b1st:
            V = np.array([img])
            b1st = False
        else:
            V = np.concatenate([V, [img]])
    return V
    #return np.array(V)

# Convert a gray image to an RGB image
# image: a single-channel image
def grayToRGB(image):
    H, W = image.shape
    return np.tile(np.reshape(image, [H, W, 1]), [1, 1, 3]) 

def reverseRGB(image):
    return np.flip(image, axis = -1)        

# Convert a [0, 1]-nomralized image to 8-bit data
# The image must be a normalized data
def toU8(image):    
    return np.array(image * 255.0, dtype = np.uint8)

# The image must be a normalized data
def inverseIntensity(image):
    return 1.0 - image


def saveImage(filepath, image, dtype = np.uint8, revChannel = False):
    if revChannel and len(image.shape) >= 3 and image.shape[2] >= 3:
        image = reverseRGB(image)    
    if(dtype == np.uint16):
        imgOut = np.uint16(image * 65535.0)
    elif(dtype == np.uint8):    
        imgOut = np.uint8(image * 255.0) 
    else:
        imgOut = image
    return cv2.imwrite(filepath, imgOut, [cv2.IMWRITE_PNG_COMPRESSION, 0])    

def saveImg(filepath, image, dtype = np.uint8, revChannel = False):
    return saveImage(filepath, image, dtype, revChannel)   

# ...............
def saveGrayImage(path, data, bits = 8):
    grayscale = 1 << 8
    dtype = np.uint8
    if bits == 16:
        dtype = np.uint16
        grayscale = 1 << 16
    elif bits == 32:
        dtype = np.uint32
        grayscale = 1 << 32
    grayscale -= 1
    im = np.array(data * grayscale, dtype = dtype)
    return cv2.imwrite(path, im)    

def saveGrayImg(path, data, bits = 8):
    return saveGrayImage(path, data, bits)     


def showImage(image, size = None, figsize = None, cmap = 'gray'):
    plt.figure(figsize=figsize)
    if size is not None:
        image = np.reshape(image, size)
    if len(image.shape) > 2:
        nH, nW, nC = image.shape
        if nC == 1:
            image = np.reshape(image, image.shape[0:2])
    else:
        nH, nW = image.shape
        nC = 1

    if nC == 1:     
        plt.imshow(image, cmap=cmap, vmin = 0, vmax = 1)
    else:
        plt.imshow(image, vmin = 0, vmax = 1)
    plt.show() 

def showImg(image, size = None, figsize = None):
    showImage(image, size, figsize= figsize)

def showImageTable(images, rows, cols, size = None, figsize = None, cmap = 'gray', caption = None):
    fig = plt.figure(figsize=figsize)
    for i, image in enumerate(images):  
        ax = fig.add_subplot(rows, cols, i + 1)
        if not (caption is None):
            ax.title.set_text(caption[i])
        if size is not None:
            image = np.reshape(image, size)
        if len(image.shape) > 2:
            nH, nW, nC = image.shape
            if nC == 1:
                image = np.reshape(image, image.shape[0:2])
        else:
            nH, nW = image.shape
            nC = 1        
        if nC == 1:     
            plt.imshow(image, cmap=cmap, vmin = 0, vmax = 1)
        else:
            plt.imshow(image, vmin = 0, vmax = 1)
    plt.show()     


def subimage(image, rect):
    return image[rect[0]:rect[1], rect[2]:rect[3]]    

def imagePaste(imageS, imageT, rect):
    imageT[rect[0]:rect[1], rect[2]:rect[3]] = imageS

def imageMSE(imageS, imageT):
    return np.linalg.norm(imageS - imageT)

# return:  ymin, ymax, xmin, xmax in Cartesian  
def imageBoundary(shape):
    nHh = shape[0] // 2
    nWh = shape[1] // 2
    boundary = (-nHh, shape[0] - nHh, -nWh, shape[1] - nWh)
    return boundary

def boundary(shape):
    shape = np.array(shape)
    min_bd = -shape // 2
    max_bd = shape + min_bd - 1
    return np.array([min_bd, max_bd])

def mostIntensity(image, bins = 256):
    rMax = bins - 1
    histo = np.histogram(image * rMax, bins=bins, range = (0.0, bins))
    iMax = np.argmax(histo[0])
    return histo[1][iMax] / rMax

# image is [H, W, C], where C must be >= 3
def mostRGB(image, bins = 256):
    return [mostIntensity(image[:, :, 0], bins = bins), 
            mostIntensity(image[:, :, 1], bins = bins),
            mostIntensity(image[:, :, 2], bins = bins) ]    

# downsampling a image (2D array) with Gaussian
def downSample(image, factor = 2, kernelsize = 5, sigma = 1.0):
    img_blur = cv2.GaussianBlur(image,(kernelsize, kernelsize), sigma)    
    return img_blur[0::factor, 0::factor]

# Resize an image (2D array)
def resize(image, factor, factorh = None, interpolation = cv2.INTER_AREA, blur = True, blursize = 5, blurstdv = 0.0):
    h, w = image.shape[0:2]
    if factorh is None:
        factorh = factor
    w_o = int(w * factor)
    h_o = int(h * factorh)
    dim = (w_o, h_o)
    # resize image
    img_o = cv2.resize(image, dim, interpolation = interpolation)
    if blur:
        img_o = cv2.GaussianBlur(img_o, (blursize, blursize), blurstdv)
    return img_o
    
# =================================================================================    

def entropy(A, bins, range = (0.0, 1.0)):
    A = np.array(A)
    H, Bins = np.histogram(A, bins = bins, range = range) 
    H = H / np.size(A)
    E = np.log2(H)
    E = np.where(np.isinf(E), 0.0, E)
    E = H * E
    return -np.sum(E)    

# aX + b
def linear2(X):
    X = np.unique(X)
    n = np.size(X)
    M = np.matrix(np.ones((n, 2)))  # 3 x 2 matrix
    I = np.reshape(np.linspace(0, n-1, n), (n, 1)) / (n - 1)
    M[:, 0] = I
    #print(np.shape(M))
    try:
        R = np.linalg.lstsq(M, X, rcond = None)
        return R[0] # [a, b]
    except:
        #return np.array([1.0, 0.0])
        return np.array([0., np.median(X)])

from pathlib import Path
def parsePath(filepath):  
    p = Path(filepath)  
    directory = str(p.parent)
    filename = str(p.stem)
    extname = str(p.suffix)
    return directory, filename, extname

def shrinkImg(image, intervalX = 1, intervalY = 1):    
    return image[::intervalY + 1, ::intervalX + 1]

def createPyramid(image, maxD = 5, minW = 0, minH = 0, intervalX = 1, intervalY = 1):
    h, w = image.shape[0:2]

    lv = 1
    pyd = [image]
    imgP = image
    while lv < maxD:
        imgP = shrinkImg(imgP, intervalX, intervalY)
        h, w = imgP.shape[0:2]
        if h >= minH and w >= minW:
            pyd.append(imgP)            
            lv += 1
        else:
            break
    return pyd

def createPyramidEx(image, maxD = 5, minW = 0, minH = 0, 
                    factor = 0.5, factorh = 0.5, interpolation = cv2.INTER_AREA, blur = True, blursize = 5, blurstdv = 0.0):
    h, w = image.shape[0:2]
    lv = 1
    pyd = [image]
    imgP = image
    if factorh is None:
        factorh = factor    
    while lv < maxD:
        w = int(w * factor)
        h = int(h * factorh)
        if h < minH or w < minW:
            break

        imgP = resize(imgP, factor, factorh, interpolation, blur, blursize, blurstdv)   
        pyd.append(imgP)            
        lv += 1
    return pyd    

def findVLines(image, thres = 0.01):
    h, w = image.shape
    VL = []
    for i in range(1, w - 1):
        I = image[:, i]       
        Ip = image[:, i - 1]            
        In = image[:, i + 1]
        dIp = I - Ip
        dIn = I - In
        tvp = (np.sum(dIp)) / h
        tvn = (np.sum(dIn)) / h
        if(np.sign(tvp) == np.sign(tvn)):
            tvp = abs(tvp)
            tvn = abs(tvn)            
            if(tvp >= thres and tvn >= thres):
                VL.append(i)    
    return VL    

def metric(A):
    return np.min(A), np.max(A), np.mean(A), np.median(A)

# Generate all pixel location in Cartesian space
# size: a 2-element tuple, (height, width)
# return: X, Y, the locations of X-axis and Y-axis, [#pixel]
def cartLoc(size):
    h, w = size
    #w2 = w // 2
    #h2 = h // 2
    #Xi = np.linspace(-w2, w2 - 1, w)
    #Yi = np.linspace(h2 - 1, -h2, h)
    #X, Y = np.meshgrid(Xi, Yi)
    X, Y = np.meshgrid(range(w), range(h))
    X = X - (w >> 1)
    Y = (h >> 1) - Y 
    return X, Y 
    
#def cart2pol(x, y):
#    rho = np.sqrt(x**2 + y**2)
#    phi = np.arctan2(y, x)
#    return(rho, phi)

#def pol2cart(rho, phi):
#    x = rho * np.cos(phi)
#    y = rho * np.sin(phi)
#    return(x, y)    

def pol2cart(r,theta):
    '''
    Parameters:
    - r: float, vector amplitude
    - theta: float, vector angle
    Returns:
    - x: float, x coord. of vector end
    - y: float, y coord. of vector end
    '''

    z = r * np.exp(1j * theta)
    x, y = z.real, z.imag

    return x, y

def cart2pol(x, y):
    '''
    Parameters:
    - x: float, x coord. of vector end
    - y: float, y coord. of vector end
    Returns:
    - r: float, vector amplitude
    - theta: float, vector angle
    '''

    z = x + y * 1j
    r,theta = np.abs(z), np.angle(z)
    return r,theta
    
# ...............
# return radius[H, W] and radian[H, W]
def polarLocation(size):
    H, W = size
    Hh = H >> 1
    Wh = W >> 1    
    LH = np.arange(start = 0.0, stop = float(H), step = 1.0) - Hh
    LW = np.arange(start = 0.0, stop = float(W), step = 1.0) - Wh    
    X, Y = np.meshgrid(LW, LH)
    radius = np.sqrt(np.square(X) + np.square(Y))
    radian = np.arctan2(Y, X)
    return radius, radian

def sigmoidExp(x):
    return np.where(x > 0.0, 1. / (1. + np.exp(-x)), np.exp(x) / (np.exp(x) + np.exp(0.0)))    
 
def sigmoid(x):    
    return .5 * (1 + np.tanh(.5 * x))    

def dsigmoid(x):
    s = sigmoid(x)
    return  s * (1.0 - s)    

# cell: [row, column]
# rect: [left, right, top, bottom]
# return: True if cell in rect; other wise, False.
def isCellInRect(cell, rect):
    return (rect[0] <= cell[0] < rect[1]) and (rect[2] <= cell[1] < rect[3])


# rectST, rectT: [left, right, top, bottom], 
#    rectST: the rect of source on the target.
#    rectT: the rect of the target.
# return rectAdjS, rectAdjT: the adjusted rect of source and target.
#        True if rect1 and rect2 have overlap; other wise, False.
def rectOnRect(rectST, rectT): 
    # Target rect
    tS, bS, lS, rS = rectST
    tT, bT, lT, rT = rectT
    height = bS - tS
    width = rS - lS      
    tAS, bAS, lAS, rAS = [0, height, 0, width]
    tAT, bAT, lAT, rAT = rectST
  
    if tS < tT:
        tAS = tT - tS   
        #bAS = tAS + height     
        tAT = tT
        #bAT = tAT + height
    if bS > bT:
        bAS -= bS - bT
        bAT = bT           
    
    if lS < lT:
        lAS = lT - lS    
        #rAS = lAS + width        
        lAT = lT        
        #bAT = lAT + width
    if rS > rT:
        rAS -= rS - rT
        rAT = rT  
    
    rectAdjS = [tAS, bAS, lAS, rAS]
    rectAdjT = [tAT, bAT, lAT, rAT]
    check = rectAdjS[0] < rectAdjS[1] and rectAdjS[2] < rectAdjS[3]
    rectAdjS[0] = np.clip(rectAdjS[0], 0, height - 1)
    rectAdjS[1] = np.clip(rectAdjS[1], 1, height)
    rectAdjS[2] = np.clip(rectAdjS[2], 0, width - 1)
    rectAdjS[3] = np.clip(rectAdjS[3], 1, width)
    return rectAdjS, rectAdjT, check

# rect1, rect2: [left, right, top, bottom]
# return: True if rect1 and rect2 have overlap; other wise, False.
def isRectOnRect(rect1, rect2):
    rectAdjS, rectAdjT, check = rectOnRect(rect1, rect2)
    return check


def rotatex(theta, m44 = False):
    cost = np.cos(theta)
    sint = np.sin(theta)
    if m44:
        return np.array([[1, 0, 0, 0], [0, cost, -sint, 0], [0, sint, cost, 0], [0, 0, 0, 1]])
    else:
        return np.array([[1, 0, 0], [0, cost, -sint], [0, sint, cost], [0, 0, 1]])
def rotatey(theta, m44 = False):
    cost = np.cos(theta)
    sint = np.sin(theta)
    if m44:
        return np.array([[cost, 0, sint, 0], [0, 1, 0, 0], [-sint, 0, cost, 0], [0, 0, 0, 1]])
    else:
        return np.array([[cost, 0, sint], [0, 1, 0], [-sint, 0, cost]])
def rotatez(theta, m44 = False):
    cost = np.cos(theta)
    sint = np.sin(theta)
    if m44:
        return np.array([[cost, -sint, 0, 0], [sint, cost, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    else:
        return np.array([[cost, -sint, 0], [sint, cost, 0], [0, 0, 1]])         
def rotate(rx, ry, rz, m44 = False):
    mtx = rotatex(rx, m44)
    mty = rotatey(ry, m44)
    mtz = rotatez(rz, m44)
    return np.matmul(mtz, np.matmul(mty, mtx))

def rotatexd(theta, m44 = False):
    cost = np.cos(theta)
    sint = np.sin(theta)
    R = None
    Rd = None
    if m44:
        R = np.array([[1, 0, 0, 0], [0, cost, -sint, 0], [0, sint, cost, 0], [0, 0, 0, 1]])
        Rd = np.array([[0, 0, 0, 0], [0, -sint, -cost, 0], [0, cost, -sint, 0], [0, 0, 0, 0]])
    else:
        R = np.array([[1, 0, 0], [0, cost, -sint], [0, sint, cost], [0, 0, 1]])
        Rd = np.array([[0, 0, 0], [0, -sint, -cost], [0, cost, -sint], [0, 0, 0]])
    return R, Rd

def rotateyd(theta, m44 = False):
    cost = np.cos(theta)
    sint = np.sin(theta)
    R = None
    Rd = None
    if m44:
        R = np.array([[cost, 0, sint, 0], [0, 1, 0, 0], [-sint, 0, cost, 0], [0, 0, 0, 1]])
        Rd = np.array([[-sint, 0, cost, 0], [0, 0, 0, 0], [-cost, 0, -sint, 0], [0, 0, 0, 0]])
    else:
        R = np.array([[cost, 0, sint], [0, 1, 0], [-sint, 0, cost]])
        Rd = np.array([[-sint, 0, cost], [0, 0, 0], [-cost, 0, -sint]])
    return R, Rd

def rotatezd(theta, m44 = False):
    cost = np.cos(theta)
    sint = np.sin(theta)
    R = None
    Rd = None
    if m44:
        R = np.array([[cost, -sint, 0, 0], [sint, cost, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        Rd = np.array([[-sint, -cost, 0, 0], [cost, -sint, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    else:
        R = np.array([[cost, -sint, 0], [sint, cost, 0], [0, 0, 1]])
        Rd = np.array([[-sint, -cost, 0], [cost, -sint, 0], [0, 0, 0]])
    return R, Rd  

def rotated(rx, ry, rz, m44 = False):
    Rx, Rdx = rotatexd(rx, m44)
    Ry, Rdy = rotateyd(ry, m44)
    Rz, Rdz = rotatezd(rz, m44)
    R = np.matmul(Rz, np.matmul(Ry, Rx))
    Rdx = np.matmul(Rz, np.matmul(Ry, Rdx))
    Rdy = np.matmul(Rz, np.matmul(Rdy, Rx))
    Rdz = np.matmul(Rdz, np.matmul(Ry, Rx))
    return R, Rdx, Rdy, Rdz

def translate(tx, ty, tz):
    return np.array([[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]])

def scale(sx, sy, sz, m44 = False):
    if m44:
        return np.array([[sx, 0, 0, 0], [0, sy, 0, 0], [0, 0, sz, 0], [0, 0, 0, 1]])
    else:
        return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, sz]]) 


def box(shape):
    sizeG = np.array([shape[2], shape[1], shape[0]])
    sizeHG = np.floor(sizeG / 2.0)
    return np.array([-sizeHG, sizeG - sizeHG, sizeG])

from matplotlib.animation import FuncAnimation
class ImageSetFig:
    imgSet = None
    pim = None
    fig = None
    animation = None
    def __init__(self, imgSet, figsize = None):
        self.imgSet = imgSet
        self.fig = plt.figure( figsize = figsize )

    def frameFunc(self, i):
        self.pim.set_array(self.imgSet[i])
        return [self.pim]

    def show(self, interval=100, aspect=None, repeat = True, repeat_delay = 0, cmap = 'gray'):
        n = len(self.imgSet)
        if n > 0:
            self.pim = plt.imshow(self.imgSet[0], aspect=aspect, vmin=0., vmax=1., cmap = cmap)
            self.animation = FuncAnimation(self.fig, func=self.frameFunc, frames=range(n), interval=interval, repeat = repeat, repeat_delay = repeat_delay)
            plt.show()
# @ ImageSetFig           

# _I is a torch.Tensor with the shape of (n_images, n_channels, height, width)
# kernel_size is a non-negative integer 
# The return is a torch.Tensor with the shape of (n_images, n_channels, n_patches, n_kernel_px)
def unfoldImage(_I, kernel_size, _Iout = None):
    n_images, n_channels, height, width = _I.shape    
    n_kernel_px = kernel_size * kernel_size
    _Iuf = F.unfold(_I, kernel_size = kernel_size).permute((0, 2, 1))
    _, n_patches, _ = _Iuf.shape # (n_images, n_channels * n_kernel_px, n_patches)
    if _Iout is None:
        _Iout = torch.empty([n_images, n_channels, n_patches, n_kernel_px], dtype = _I.dtype, device = _I.device)
    i_ps = 0
    i_pe = n_kernel_px
    for i in range(n_channels):
        _Iout[:, i, :, :] = _Iuf[:, :, i_ps:i_pe]
        i_ps = i_pe
        i_pe += n_kernel_px
    return _Iout # (n_images, n_channels, n_patches, n_kernel_px)
 
 # _I is a torch.Tensor with the shape of (n_images, n_channels, height, width)
# pad_size is a non-negative integer 
# The return is a torch.Tensor with the shape of (n_images, n_channels, n_patches, 1)   
def unfoldCenterImage(_I, pad_size = 0):
    if pad_size > 0:
        return _I[:, :, pad_size:-pad_size, pad_size:-pad_size].flatten(start_dim = 2).unsqueeze(3)    
    else:
        return _I.flatten(start_dim = 2).unsqueeze(3)


def clone(_tensor):
    return _tensor.detach().clone()

def toNumpy(_tensor):
    return _tensor.detach().cpu().numpy()  

# Convert (n, c, h, w) to (n, h, w, c)
def toNumpyImage(_tensor, normalize = False):
    img = _tensor.permute(0, 2, 3, 1).detach().cpu().numpy() #(n, h, w, channels)
    if normalize:
        img = vgi.normalize(img)
    return img

# Convert (h, w), (h, w, c), or (n, h, w, c) to (n, c, h, w)
def toTorchImage(image, dtype = torch.float, device = None):
    n_dim = len(image.shape)
    _I = torch.tensor(image, dtype = torch.float, device = device)
    if n_dim == 2: #(h, w)
        _I = _I.unsqueeze(0).unsqueeze(0)
    elif n_dim == 3: #(h, w, channels)
        _I = _I.permute(2, 0, 1).unsqueeze(0)
    elif n_dim == 4: #(n, h, w, channels)
        _I = _I.permute(0, 3, 1, 2)
    return _I     

def showTorchImage(_I, i = 0, nml = True, figsize = None):
    if nml:
        showImage(normalize(toNumpyImage(_I)[i]), figsize = figsize)
    else:
        showImage(toNumpyImage(_I)[i], figsize = figsize)    

def evaluateImage(I, target, data_range = None):
    nC = 1
    if len(target.shape) > 2:
        nC = target.shape[-1]
    else:
        nc = 1
        target = np.expand_dims(target, -1)
        I = np.reshape(I, target.shape)
    diff = I - target
    v_mae = np.mean(np.abs(diff))
    v_mse = mse(I, target) 
    v_ssim = 0.0
    v_psnr = 0.0
    for i in range(nC):
        Ic = I[..., i]
        Tc = target[..., i]
        if data_range is None:
            data_range = Tc.max() - Tc.min()
        v_ssim += ssim(Ic, Tc, data_range = data_range)
        v_psnr += psnr(Ic, Tc, data_range = data_range)
    v_ssim /= nC
    v_psnr /= nC
    return v_mae, v_mse, v_ssim, v_psnr       

