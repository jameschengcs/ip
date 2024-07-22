'''
FSP v1.0
# (c) 2024, Chang-Chieh Cheng, jameschengcs@nycu.edu.tw
'''
import numpy as np
import torch
import time
import random
import json
import cv2 
import vgi
import cc3d
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchvision.transforms import ToTensor, GaussianBlur
from operator import itemgetter

# Coordinate system:
#                    | +y = (h // 2)
#                    |
# -x = -(w // 2) --- 0 --- +x < (w // 2)
#                    |
#                    | -y >= -(h // 2)
  
img2Tensor = ToTensor()
     
ARGN = 9  # mu_x, mu_y, sigma_x, sigma_y, theta, alpha, beta_r, beta_g, beta_b
          # 0     1     2        3         4     5      6       7       8
# Samples are uniformly distributed over the half-open interval [low, high)
# return [n, ARGN]
def randArg(argMin, argMax, n = 1):
    return np.random.uniform(argMin, argMax, [n, ARGN]) 

def clone(_tensor):
    return _tensor.detach().clone()

def toNumpy(_tensor):
    return _tensor.detach().cpu().numpy()  

def toNumpyImage(_tensor, normalize = False):
    img = _tensor.permute(0, 2, 3, 1)[0].detach().cpu().numpy()
    if normalize:
        img = vgi.normalize(img)
    return img
# 4 bytes * 3 channels * 9 dimentions * 10 batch_size * 7 * 7 patch_size * 256 * 256 pixels
# 108 * 10 batch_size * 7 * 7 patch_size * 256 * 256 pixels
# 1080 * 7 * 7 patch_size * 256 * 256 pixels
# 1080 * 7 * 7 patch_size * 256 * 256 pixels
# 52920 * 256 * 256 pixels
# 3,468,165,120 ==> 3.23 GB

# Check that a circle, Cs, overlaps n circles, Ct1, Ct2, ... Ctn.
# This function can be used for numpy and pytorch arrays.
# rs + rt >= d(s, t)
# xs(1,) and ys(1,): center of Cs.
# rs(1,): the radius of the Cs.
# xt(n,) and yt(n,): centers of Ct.
# rt(n,): the radii of Ct.
# Return: True, if Cs overlaps any Ct; False, otherwise.
def circlex(xs, ys, rs, xt, yt, rt):
    dx = (xs - xt) # (n, )
    dy = (ys - yt) # (n, )
    d = (dx * dx + dy * dy) ** 0.5 # (n, )
    r = rs + rt # (n, )
    M = (r >= d) # (n,)
    return M.any() # numpy and torch support that.

def bbsize2d(bb):
    n = 1
    shape = []
    for slc in bb:
        if slc is None:
            continue
        d = slc.stop - slc.start
        n *= d
        shape += [d]
    return n, shape      

class painter:
    # All torch.tensor of images must be packed as (batch_size, channels, height, width)
    def tensor(self, data, dtype = torch.float):
        return torch.tensor(data, dtype = dtype, device = self.device)
    def zeros(self, shape, dtype = torch.float):
        return torch.zeros(shape, dtype = dtype, device = self.device)
    def ones(self, shape, dtype = torch.float):
        return torch.ones(shape, dtype = dtype, device = self.device)   
    def full(self, shape, value, dtype = torch.float):
        return torch.full(shape, value, dtype = dtype, device = self.device)   
    def validarg(self, g):
        _g = g if torch.is_tensor(g) else self.tensor(g)
        n = torch.numel(_g)
        _g = _g.reshape([n // ARGN, ARGN]) 
        return _g

    
    def __init__(self, target, arg = None, 
                    bg = None, prev_image = None, gpu = True,  
                    pyd_max_levels = 4, pyd_min_size = 128, max_alpha = 0.9,
                    search_batch_size = 8, search_primitives = 4, area_f = 1.5,
                    fix_sx = False, fix_theta = False, keep_aspect = False,
                    min_size = 1.0, loss_ratio = 0.5):
        self.debug = False
        self.loss_count = 0
        self._debugData = None
        self.gamma = 180.0

        self.loss_reduce_axis = (-3, -2, -1)

        if gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.gpu = True
        else:
            self.device = torch.device("cpu")
            self.gpu = False

        self.longTensor = torch.cuda.LongTensor if self.gpu else torch.LongTensor
        self.pca = PCA()
        self.bg = bg
        if self.bg is None:
            bg_bins = 256
            self.bg = vgi.mostRGB(target, bins = bg_bins)
                        
        self._bg = self.tensor(self.bg)      
        self._exp_thres = self.tensor(-15.0)
        self._one = self.tensor(1.0)    
        self._zero = self.tensor(0.0) 
        self._largest_loss = self.tensor(999999999999.0) 
        self._sigmoid = torch.nn.Sigmoid()        
        self._sigmoid_thres = self.tensor(10.0) 
        self._sigmoid_thresinv = -self._sigmoid_thres    
        self.min_size = min_size       
        self._step_size = torch.tensor( [0.5, 0.5, 0.5, 0.5, np.pi / 180., 
                                        1/256.0, 1/256.0, 1/256., 1/256], device = self.device)              
        self.loss_ratio = loss_ratio
        self._weight = None
        self._mse = torch.nn.MSELoss()
        self._L1 = torch.nn.L1Loss()

        self.max_alpha = max_alpha
        self.search_batch_size = search_batch_size
        self.search_primitives = search_primitives
        self.search_n = search_batch_size * self.search_primitives
        self._search_idx_acc = torch.arange(self.search_batch_size, dtype = torch.int64, device = self.device) * self.search_primitives
        self.area_f = area_f
        self.brush_masking = True
        self.brush_antialiasing = False
        self.fix_sx = False if fix_sx is None else fix_sx
        self.fix_theta = False if fix_theta is None else fix_theta
        self.keep_aspect = False if keep_aspect is None else keep_aspect
        self.stroke_aspect = None

        self.target = target
        self.channels = self.target.shape[-1] # (h, w, c)
        self.pyd_max_levels = pyd_max_levels
        self.pyd_min_size = pyd_min_size
        self.target_pyd, self._target = self.createPyramid(self.target, self.pyd_max_levels, self.pyd_min_size)
        self.pyd_levels = len(self._target)

        self._A = [None] *  self.pyd_levels   # [levels, (nQ, ARGN)], argument set
        if not (arg is None):
            self.setInitalArg(arg, lv = 0)        

        self.shape = []
        self.pixels = []
        self.boundary = []
        self._Y = []
        self._X = []
        self.arg_min = []
        self.arg_max = []
 
        if prev_image is None:
            self._I = []
        else:
            _, self._I = self.createPyramid(prev_image, self.pyd_max_levels, self.pyd_min_size)       
            
        for lv in range(self.pyd_levels):
            _target_lv = self._target[lv]
            _, _, h_lv, w_lv = _target_lv.shape
            self.shape += [_target_lv.shape]
            self.pixels += [h_lv * w_lv]  
            boundary_lv = vgi.imageBoundary((h_lv, w_lv))
            self.boundary += [boundary_lv]  # (ymin, ymax, xmin, xmax) in Cartesian  
            arg_min_lv = np.array([-w_lv / 2.0, -h_lv / 2.0, self.min_size, self.min_size, -np.pi, 0.01, 0.0, 0.0, 0.0])
            arg_max_lv = np.array([ w_lv / 2.0,  h_lv / 2.0, w_lv, h_lv, np.pi, max_alpha, 1.0, 1.0, 1.0])
            self.arg_min += [arg_min_lv]
            self.arg_max += [arg_max_lv]

            # _Y_lv, _X_lv: (h, w)
            _Y_lv, _X_lv = torch.meshgrid(torch.arange(-boundary_lv[0], -boundary_lv[1], -1, device = self.device), 
                                          torch.arange( boundary_lv[2],  boundary_lv[3], device = self.device),
                                          indexing = 'ij')
            self._Y += [_Y_lv]
            self._X += [_X_lv]

            if prev_image is None:
                _I_lv = self._bg.repeat((h_lv, w_lv, 1)).permute(2, 0, 1).unsqueeze(0)
                self._I += [_I_lv]

        # level loop
    # painter::__init__

    # ....................................
    # Reconstruced images

    def background(self):
        if self._bg is None:
            return np.array([0.0, 0.0, 0.0])
        else:
            return toNumpy(self._bg)

    def image(self, lv = 0, BHWC = True):
        _I = self._I[lv] 
        if BHWC:
            # self._I is (batch_size, channels, height, width)
            _I = _I.permute(0, 2, 3, 1)                 
            if(_I.shape[0] == 1):
                return toNumpy(_I[0])
            else:
                return toNumpy(_I)
        else:
            return toNumpy(_I)
    # painter::image

    # image torch.tensor
    def toImageTensor(self, image):
        _I = self.tensor(image).moveaxis(-1, -3)
        if len(image.shape) < 4:
            _I = _I.unsqueeze(0)
        return _I

    # Pyramid creating
    def createPyramid(self, image, max_levels = 5, min_size = 128):
        img_pyd = vgi.createPyramidEx(image, maxD = max_levels, minW = min_size, minH = min_size)
        levels = len(img_pyd)
        _img_pyd = []
        for img in img_pyd:
            _img_pyd += [self.toImageTensor(img)]
        return img_pyd, _img_pyd

    def json(self, lv = 0):
        argSet = self.argument(lv)
        bg = self.background()
        n, c, h, w =  self.shape[lv]
        jsonOut = {'shape':(h, w, c),
                   'bg': None if bg is None else bg.tolist(), 
                   'Q': None if argSet is None else argSet.tolist()}           
        return jsonOut
    # painter::json

    def save(self, image_path = None, json_path = None, lv = 0): 
        I = self.image(lv = lv)   
        rt = I   
        if not(image_path is None):  
            vgi.saveImage(image_path, I, revChannel = True)

        json_out = None
        if not(json_path is None):    
            json_out = self.json(lv = lv)        
            with open(json_path, 'w') as jfile:
                json.dump(json_out, jfile) 
            rt = (I, json_out)
        return rt
    # painter::save        

    # ....................................
    # Geometry

    # loc:(n, 2), n * (row, column) 
    # return:(n, 2), n * (x, y)
    # formula: x = loc - wh; 
    #          y = hh - row. 
    def ij2xy(self, loc, lv = 0):
        row =  loc[..., 0:1]
        col =  loc[..., 1:]
        # self.boundary: -hh, hh, -wh, wh; e.g., (-128, 128, -128, 128) for (256, 256) 
        #                                        and (-128, 129, -128, 128) for (257, 256)
        x = col + self.boundary[lv][2]
        y = -self.boundary[lv][0] - row 
        return np.concatenate([x, y], axis = -1)

    # p:(n, 2), n * (x, y) 
    # return:(n, 2), n * (row, column)
    # formula: row = hh - y; 
    #          col = x + wh.
    def xy2ij(self, p, lv = 0, round_int = False):
        x =  p[..., 0:1]
        y =  p[..., 1:]
        # self.boundary: -hh, hh, -wh, wh; e.g., (-128, 128, -128, 128) for (256, 256) 
        #                                        and (-128, 129, -128, 128) for (257, 256)
        row = -self.boundary[lv][0] - y 
        col = x - self.boundary[lv][2]

        idx = np.concatenate([row, col], axis = -1)
        if round_int:
            idx = np.round(idx).astype(np.int32)            
        return idx

    def X(self, lv = 0):
        return toNumpy(self._X[lv])
    def Y(self, lv = 0):
        return toNumpy(self._Y[lv])



    # ............................................ 
    # Primtive arguments
    def argument(self, lv = 0):
        if self._A[lv] is None:
            return None
        else:                    
            arg = toNumpy(self._A[lv])
            return arg

    def setArg(self, arg, lv = 0):
        nQ = arg.shape[0]
        self._A[lv] = self.tensor(arg.reshape([nQ, ARGN]))
    # painter::setArg   
    
    # _arg: (primtives, ARGN)
    def appendArg(self, _arg, lv = 0):
        if self._A[lv] is None:
            self._A[lv] = _arg
        else:
            self._A[lv] = torch.cat( (self._A[lv], _arg) )
    # painter::appendArg

    def scaleArg(self, _A, x, y = None):
        if y is None:
            y = x
        _A_out = clone(_A) if torch.is_tensor(_A) else np.array(_A)
        _A_out[:, 0] *= x # ux
        _A_out[:, 1] *= y # uy
        _A_out[:, 2] *= x # sx
        _A_out[:, 3] *= y # sy
        return _A_out

    def downscaleArg(self, lv):
        lv_next = lv + 1
        if lv_next >= self.pyd_levels:
            return self._A[lv]            
        self._A[lv_next] = self.scaleArg(self._A[lv], 0.5)  
        return self._A[lv_next]       

    def upscaleArg(self, lv):        
        if lv == 0:
            return self._A[lv]   
        lv_prev = lv - 1         
        self._A[lv_prev] = self.scaleArg(self._A[lv], 2.0)  
        return self._A[lv_prev]  

    def randomInitArg(self, n, lv = 0, factor_u = 0.5, factor_s = 0.125, shrink_rate = 0.9, shrink_min = 0.1, shrink_n = None, min_size = 1.0):
        _, _, height, width = self.shape[lv]

        x_max = width * factor_u
        y_max = height * factor_u
        x_min = -x_max
        y_min = -y_max
        witdh_max = width * factor_s
        height_max = height * factor_s
        witdh_min = min_size
        height_min = min_size
        theta_max = np.pi / 2.0
        theta_min = -theta_max
        alpha_max = 0.8
        alpha_min = 0.8
        beta_max = 1.0
        beta_min = 0.0

        argRandMin = np.array([x_min, y_min, witdh_min, height_min, theta_min, alpha_min, beta_min, beta_min, beta_min])
        argRandMax = np.array([x_max, y_max, witdh_max, height_max, theta_max, alpha_max, beta_max, beta_max, beta_max])
        remain = n
        arg = None
        shrink = True
        if shrink_n is None:
            shrink_n = n
            shrink = False
        while remain > 0:
            if remain < shrink_n:
                shrink_n = remain
            argB = randArg(argRandMin, argRandMax, shrink_n)             

            if arg is None:
                arg = argB
            else:
                arg = np.concatenate((arg, argB))
            remain -= shrink_n
            if shrink:
                factor_s  *= shrink_rate
                if factor_s < shrink_min:
                    factor_s = shrink_min
                    shrink = False
                argRandMax[2] = width * factor_s
                argRandMax[3] = height * factor_s
                if argRandMax[2] <= argRandMin[2]:
                    argRandMax[2] = argRandMin[2]
                if argRandMax[3] <= argRandMin[3]:
                    argRandMax[3] = argRandMin[3]
        return arg                                                 
    # painter::randomInitArg        

    # ...........................................
    # loss functions
    # _I: (n, c, h, w)
    # return: (n, )
    def lossMSE(self, _I, lv = 0):       
        #_d = ((_I - self._target[lv]) ** 2.0)             
        _d = _I - self._target[lv] 
        _d *= _d       
        _loss = _d.mean(dim = self.loss_reduce_axis).sqrt()
        return _loss           

    def lossL1(self, _I, lv = 0):
        _d = ((_I - self._target[lv]).abs())            
        _loss = _d.mean(dim = self.loss_reduce_axis)
        return _loss    

    def lossL1L2(self, _I, lv = 0):
        _lossL1 = self.lossL1(_I, lv = lv)
        _lossL2 = self.lossMSE(_I, lv = lv)
        _loss = self.loss_ratio * _lossL1 + (1 - self.loss_ratio) * _lossL2
        self.loss_count += 1
        return _loss                    
  
    def getLoss(self, loss = 'maemse'):
        lf = None
        if loss == 'maemse':
            lf = self.lossL1L2     
        elif loss == 'mse':
            lf = self.lossMSE            
        elif loss == 'mae':
            lf = self.lossL1                  
        return lf   

    # ...........................................
    # Reconstruction

    # _A: a set of argument vectors of primitives, (primitives, ARGN)    
    def primitiveMeta(self, _arg, lv = 0):
        images, channels, height, width = self.shape[lv]
        pixels = self.pixels[lv]
        primitives = _arg.shape[0]

        # (primitives, )
        _ux    = _arg[:, 0:1, None, None] # (n, 1, 1, 1)
        _uy    = _arg[:, 1:2, None, None] 
        _sx    = _arg[:, 2:3, None, None]
        _sy    = _arg[:, 3:4, None, None]
        _theta = _arg[:, 4:5, None, None]
        _alpha = _arg[:, 5:6, None, None]  
        _beta  = _arg[:, 6: , None, None] # (n, 3, 1, 1)

        _sxsx  = torch.square(_sx) 
        _sysy  = torch.square(_sy)         
        _cosT  = torch.cos(_theta) 
        _sinT  = torch.sin(_theta) 

        _X = self._X[lv] # (h, w)
        _Y = self._Y[lv]
    
        _Xu = _X - _ux  # (n, 1, h, w) 
        _Yu = _Y - _uy        
        _Xt = _Xu * _cosT - _Yu * _sinT 
        _Yt = _Xu * _sinT + _Yu * _cosT  

        return _Xt, _Yt, _sx, _sy, _sxsx, _sysy, _sinT, _cosT, _alpha, _beta


    # _A: a set of argument vectors of primitives, (batch_size, dimensions, 1)    
    # returns: 
    #  _I_Ga: (primitives, channels, patch_height, patch_width) if is_unisize; otherwise, [primitives, (channels, patch_height, patch_width)]; the images of all Gaussians, alpha * G
    #  _f: (primitives, 1, patch_height, patch_width), 1 - alpha * Q    
    #  _G: (primitives, 1, patch_height, patch_width)
    def drawreturn(self, _G, _alpha, _beta, weight_only, f_only):
        ret = tuple()
        if weight_only:
            ret += (_G,)
        else: 
            _Ga = _G * _alpha    # (n, 1, h, w) = (n, 1, h, w) * (n, 1, 1, 1)
            _f = self._one - _Ga # (n, 1, h, w)
            if f_only:                
                ret += (_f,)                    
            else:
                _I_Ga = _Ga * _beta  # (n, c, h, w) = (n, 1, h, w) * (n, c, 1, 1)
                ret += (_I_Ga, _f, _G)
        return ret        

    def drawGaussian(self, _arg, lv = 0, weight_only = False, f_only = False):        
        _Xt, _Yt, _sx, _sy, _sxsx, _sysy, _sinT, _cosT, _alpha, _beta = self.primitiveMeta(_arg, lv)
        primitives = _sx.shape[0]
        _2sxsx = _sxsx + _sxsx
        _2sysy = _sysy + _sysy
 
        _XtXt = _Xt * _Xt # (n, 1, h, w) 
        _YtYt = _Yt * _Yt
        _Z = -(_XtXt / _2sxsx + _YtYt / _2sysy) # (n, 1, h, w)
        _G = torch.where(_Z > self._exp_thres, torch.exp(_Z), self._zero) # (n, 1, h, w)

        return self.drawreturn(_G, _alpha, _beta, weight_only, f_only) 
    # painter::drawGaussian
    
    # returns: 
    #  _I_Ga: [primitives, channels, pixels], the images of all Gaussians, alpha * Q
    #  _f: [primitives, pixels], 1 - alpha * Q    
    def drawEllipse(self, _arg, lv = 0, weight_only = False, f_only = False):
        _Xt, _Yt, _sx, _sy, _sxsx, _sysy, _sinT, _cosT, _alpha, _beta = self.primitiveMeta(_arg, lv)
        primitives = _sx.shape[0]
        _2sxsx = _sxsx + _sxsx
        _2sysy = _sysy + _sysy
        scalef = 1.414

        _Z = (self._one -(torch.square(_Xt) / _2sxsx + torch.square(_Yt) / _2sysy)) * self.gamma             
        _G = torch.where(torch.logical_and(_Z < self._sigmoid_thres, _Z > self._sigmoid_thresinv), self._sigmoid(_Z), _Z)
        _G = torch.where(_G >= self._sigmoid_thres, self._one, _G)
        _G = torch.where(_G <= self._sigmoid_thresinv, self._zero, _G)  # (n, 1, h, w)
        return self.drawreturn(_G, _alpha, _beta, weight_only, f_only) 
    # painter::drawEllipse    

    def drawRect(self, _arg, lv = 0, weight_only = False, f_only = False):
        _Xt, _Yt, _sx, _sy, _sxsx, _sysy, _sinT, _cosT, _alpha, _beta = self.primitiveMeta(_arg, lv)
        primitives = _sx.shape[0]
        #_gamma = self.full(_alpha.shape, self.gamma)
        scalef = 1.414
        _Zx = (self._one - torch.abs(_Xt) / (_sx * scalef)) * self.gamma
        _Zy = (self._one - torch.abs(_Yt) / (_sy * scalef)) * self.gamma

        _Qx = torch.where(torch.logical_and(_Zx < self._sigmoid_thres, _Zx > self._sigmoid_thresinv), self._sigmoid(_Zx), _Zx)
        _Qx = torch.where(_Qx >= self._sigmoid_thres, self._one, _Qx)
        _Qx = torch.where(_Qx <= self._sigmoid_thresinv, self._zero, _Qx)

        _Qy = torch.where(torch.logical_and(_Zy < self._sigmoid_thres, _Zy > self._sigmoid_thresinv), self._sigmoid(_Zy), _Zy)
        _Qy = torch.where(_Qy >= self._sigmoid_thres, self._one, _Qy)
        _Qy = torch.where(_Qy <= self._sigmoid_thresinv, self._zero, _Qy)   
        _G = _Qx * _Qy   

        return self.drawreturn(_G, _alpha, _beta, weight_only, f_only)   
    # painter::drawRect  

    def drawRectB(self, _arg, lv = 0, weight_only = False, f_only = False):
        _Xt, _Yt, _sx, _sy, _sxsx, _sysy, _sinT, _cosT, _alpha, _beta = self.primitiveMeta(_arg, lv)
        primitives = _sx.shape[0]
        #_gamma = self.full(_alpha.shape, self.gamma)
        scalef = 1.0
        _Zx = (self._one - torch.abs(_Xt) / (_sx * scalef)) * self.gamma
        _Zy = (self._one - torch.abs(_Yt) / (_sy * scalef)) * self.gamma

        _Qx = torch.where(torch.logical_and(_Zx < self._sigmoid_thres, _Zx > self._sigmoid_thresinv), self._sigmoid(_Zx), _Zx)
        _Qx = torch.where(_Qx >= self._sigmoid_thres, self._one, _Qx)
        _Qx = torch.where(_Qx <= self._sigmoid_thresinv, self._zero, _Qx)

        _Qy = torch.where(torch.logical_and(_Zy < self._sigmoid_thres, _Zy > self._sigmoid_thresinv), self._sigmoid(_Zy), _Zy)
        _Qy = torch.where(_Qy >= self._sigmoid_thres, self._one, _Qy)
        _Qy = torch.where(_Qy <= self._sigmoid_thresinv, self._zero, _Qy)   
        _G = _Qx * _Qy   

        bolder_size = 1.
        _Zx = (self._one - torch.abs(_Xt) / (_sx * scalef - bolder_size)) * self.gamma
        _Zy = (self._one - torch.abs(_Yt) / (_sy * scalef - bolder_size)) * self.gamma

        _Qx = torch.where(torch.logical_and(_Zx < self._sigmoid_thres, _Zx > self._sigmoid_thresinv), self._sigmoid(_Zx), _Zx)
        _Qx = torch.where(_Qx >= self._sigmoid_thres, self._one, _Qx)
        _Qx = torch.where(_Qx <= self._sigmoid_thresinv, self._zero, _Qx)

        _Qy = torch.where(torch.logical_and(_Zy < self._sigmoid_thres, _Zy > self._sigmoid_thresinv), self._sigmoid(_Zy), _Zy)
        _Qy = torch.where(_Qy >= self._sigmoid_thres, self._one, _Qy)
        _Qy = torch.where(_Qy <= self._sigmoid_thresinv, self._zero, _Qy)   
        _G_in = 1.0 - _Qx * _Qy
        _G = _G * _G_in

        return self.drawreturn(_G, _alpha, _beta, weight_only, f_only)   
    # painter::drawRectB      

    def drawBrush(self, _brush, _arg, lv = 0, weight_only = False, f_only = False):
        _Xt, _Yt, _sx, _sy, _sxsx, _sysy, _sinT, _cosT, _alpha, _beta = self.primitiveMeta(_arg, lv)
        primitives = _sx.shape[0]
        _2sx = _sx + _sx
        _2sy = _sy + _sy
        bH, bW = _brush.shape[0:2]  

        _Zx = _Xt / _2sx
        _Zy = _Yt / _2sy
       
        _ZxA = _Zx.abs() <= 1.0
        _ZyA = _Zy.abs() <= 1.0
        _Zxi = ((_Zx + 1.0) * ((bW - 1) / 2))
        _Zyi = ((1.0 - _Zy) * ((bH - 1) / 2))
        if self.brush_antialiasing:
            _Zxif = _Zxi.floor()
            _Zxic = _Zxi.ceil()
            _Zyif = _Zyi.floor()
            _Zyic = _Zyi.ceil()     

            _Zx_r = _Zxi - _Zxif
            _Zx_rr = (1 - _Zx_r)
            _Zy_r = _Zyi - _Zyif 
            _Zy_rr = (1 - _Zy_r)

            _Zxif = _Zxif.type(self.longTensor)
            _Zxic = _Zxic.type(self.longTensor)
            _Zyif = _Zyif.type(self.longTensor)
            _Zyic = _Zyic.type(self.longTensor)


            _Zxif = torch.where(_ZxA, _Zxif, 0)
            _Zxic = torch.where(_ZxA, _Zxic, 0)   
            _Zyif = torch.where(_ZyA, _Zyif, 0)
            _Zyic = torch.where(_ZyA, _Zyic, 0)  

            _ZT = torch.logical_and(_ZxA, _ZyA)

            _Gxfyf = torch.where(_ZT, _brush[_Zyif, _Zxif], self._zero) 
            _Gxcyf = torch.where(_ZT, _brush[_Zyif, _Zxic], self._zero) 
            _Gxfyc = torch.where(_ZT, _brush[_Zyic, _Zxif], self._zero) 
            _Gxcyc = torch.where(_ZT, _brush[_Zyic, _Zxic], self._zero) 

            _Gyf = _Gxfyf * _Zx_rr + _Gxcyf * _Zx_r
            _Gyc = _Gxfyf * _Zx_rr + _Gxcyf * _Zx_r

            _G = _Gyf * _Zy_rr + _Gyc * _Zy_r

        else:
            _Zxi = _Zxi.type(self.longTensor)
            _Zyi = _Zyi.type(self.longTensor)
            _Zxi = torch.where(_ZxA, _Zxi, 0)   
            _Zyi = torch.where(_ZyA, _Zyi, 0)   
            _G = torch.where(torch.logical_and(_ZxA, _ZyA), _brush[_Zyi, _Zxi], self._zero)
        
        ret = tuple()
        if weight_only:
            ret += (_G,)
        else: 
            _Ga = _G * _alpha    # (n, 1, h, w) = (n, 1, h, w) * (n, 1, 1, 1)
            if self.brush_masking:
                _f = torch.where(_G > self._zero, self._one - _alpha, self._one) # (n, 1, h, w)
            else:
                _f = self._one - _Ga
            #_Ga_mask = _G_mask * _alpha
            #_f = self._one - _Ga_mask # (n, 1, h, w)
            if f_only:                
                ret += (_f,)                    
            else:
                _I_Ga = _Ga * _beta  # (n, c, h, w) = (n, 1, h, w) * (n, c, 1, 1)
                ret += (_I_Ga, _f, _G)
        return ret     
    # painter::drawBrush  

    def loadBrush(self, path, inv = False):
        imgBrush = vgi.loadImage(path, gray = True)   
        if inv: 
            imgBrush = 1.0 - imgBrush
        imgBrush = imgBrush.reshape(imgBrush.shape[0:2])  # only (h, w)
        self._brush0 = self.tensor(imgBrush)
        bh, bw = imgBrush.shape
        self.stroke_aspect = bh / bw #  y = xf
        #print('loadBrush::self.stroke_aspect', self.stroke_aspect)
        return self._brush0

    def drawBrush0(self, _arg, lv = 0, weight_only = False, f_only = False):      
        return self.drawBrush(_brush = self._brush0, _arg = _arg, lv = lv, weight_only = weight_only, f_only = f_only)    

    def composite(self, _arg = None, lv = 0, primitive = 'brush0', batch_size = 10, update_I = True, no_prev = False):
        if _arg is None:
            _arg = self._A[lv]
        n = _arg.shape[0]
        if no_prev:
            _I = self.zeros(self.shape[lv])
        else:
            _I = clone(self._I[lv])

        draw = self.getDrawMethod(primitive)
        i = 0
        while i < n:
            j = min(i + batch_size, n)
            nb = j - i
            _arg_batch = _arg[i:j]        
            _IG, _f, _ = draw(_arg_batch, lv = lv)                         
            for k in range(nb):
                _I =  _IG[k] + _f[k] * _I 
            i += nb
        if update_I:
            self._I[lv] = _I
        return _I
    # painter::composite
 
    def compositeWeightOnly(self, _arg = None, lv = 0, primitive = 'brush0', batch_size = 10, update_I = True, no_prev = False):
        if _arg is None:
            _arg = self._A[lv]        
        n = _arg.shape[0]

        if no_prev:
            _I = self.zeros(self.shape[lv])
        else:
            _I = clone(self._I[lv])

        draw = self.getDrawMethod(primitive)
        i = 0
        while i < n:
            j = min(i + batch_size, n)
            nb = j - i
            _arg_batch = _arg[i:j]        
            _G = self.draw(_arg, lv = lv, weight_only = True)                       
            _I +=  _G.sum(dim = 0)
            i += nb
        if update_I:
            self._I[lv] = _I
        return _I    
    # painter::compositeWeightOnly   

    def getDrawMethod(self, primitive = 'Gaussian') :
        drawShape = None
        if primitive == 'Gaussian':
            drawShape = self.drawGaussian
        elif primitive == 'ellipse':
            drawShape = self.drawEllipse
        elif primitive == 'rect':
            drawShape = self.drawRect    
        elif primitive == 'rectb':
            drawShape = self.drawRectB               
        elif primitive == 'brush' or primitive == 'brush0':
            drawShape = self.drawBrush0         
        else:
            drawShape = self.drawBrush0            
        return drawShape   
    
    # _arg: (primitives, AGRN)
    # _I: (n=1, c, h, w)
    # draw: see self.getDrawMethod()
    # lf: see self.getLoss()
    # lv: level
    # _foreground: None or (_I_fg, _f_kp)
    # return:
    # _loss: (primitives,)
    # _I_out: (primitives, c, h, w) 
    def lossList(self, _arg, _I, draw, lf, lv = 0, _foreground = None):
        primitives = _arg.shape[0] 
        _I_Ga, _f, _G = draw(_arg, lv = lv)        
        #  _I_Ga: [primitives, (c, h, w)], the images of all Gaussians, alpha * Q
        #  _f: [primitives, (1, h, w)], 1 - alpha * Q
        #  _G: [primitives, (1, h, w)]         

        # Backward alpha compositing: I_k  = \beta_k \alpha_k G_k + (1 - \alpha_k G_k)I_{k-1}
        # Forward alpha compositing: J_k = J_{k - 1} + \beta_k \alpha_k G_k * f_{k - 1}
        _I_all = (_I_Ga + _f * _I)

        if not(_foreground is None):
            #print('bestSearch::_I_all_px bg', torch.min(_I_all_px), torch.max(_I_all_px))
            _I_fg, _f_kp = _foreground # (n=1, c, h, w), (n=1, 1, h, w)
            _I_all = _I_fg + _f_kp * _I_all 
            # _I_all: (primitives, c, h, w)      

        _loss = lf(_I_all, lv = lv)
        return _loss, _I_all
    # painter::lossList

    # _arg: (primitives, AGRN)
    # _I: (n=1, c, h, w) 
    # draw: see self.getDrawMethod()
    # lf: see self.getLoss()
    # lv: level    
    # _foreground: None or (_I_fg, _f_kp)    
    # return:
    # i_best: The index of the best primtive argument
    # _I_best: (1, c, h, w)
    # _min_loss
    def bestSearch(self, _arg, _I, lf, draw, lv = 0, _foreground = None):     
        _loss, _I_all = self.lossList(_arg, _I, draw, lf, lv = lv, _foreground = _foreground)
        i_best = torch.argmin(_loss)
        _min_loss = _loss[i_best]
        _I_best = _I_all[i_best].unsqueeze(0) #(n=1, c, h, w)
        return i_best, _I_best, _min_loss
    # painter::bestSearch

    # _arg_set: (batches, primitives, AGRN)
    # _I: (1, c, h, w) 
    # draw: see self.getDrawMethod()
    # lf: see self.getLoss()
    # lv: level    
    # _foreground: None or (_I_fg, _f_kp)
    # _step: None or (batches, primitives)
    # return:
    # _arg_best: (batches, ARGN)
    # _I_best: (batches, c, h, w)
    # _loss_best: (batches, )
    # _step_best: (batches,)
    # _i_best: (batches, ), The index of the best primtive argument of each batch
    def bestSearchBatch(self, _arg_set, _I, lf, draw, lv = 0, _foreground = None, _step = None):     
        batches, primitives, _ = _arg_set.shape 
        if batches != self.search_batch_size or primitives != self.search_primitives:
            n = batches * primitives
            _i_p = torch.arange(batches, dtype = torch.int64, device = self.device) * primitives
        else:
            n = self.search_n
            _i_p = self._search_idx_acc

        _arg = _arg_set.reshape([n, ARGN])        
        _loss, _I_all = self.lossList(_arg, _I, draw, lf, lv = lv, _foreground = _foreground)
            # _loss: (n,)
            # _I_all: (n, c, h, w)
        _loss = _loss.reshape([batches, primitives])
        _loss_best, _i_best = _loss.min(dim = 1) # (batches,)
        
        _i_best_p = _i_best + _i_p

        _arg_best = _arg[_i_best_p]
        _I_best = _I_all[_i_best_p] #(batches, c, h, w)
        if _step is None:
            return _arg_best, _I_best, _loss_best, None, _i_best
        else:
            _step_p = _step.reshape([n,])
            _step_best = _step_p[_i_best_p] # (batches,)
            return _arg_best, _I_best, _loss_best, _step_best, _i_best
    # painter::bestSearchBatch    

    # ***************
    # _arg: (primitives = 1, dimensions),  _current_err [1] must be torch.tensor
    # _I: (n=1, c, h, w) 
    # _arg_min, _arg_max: (ARGN,)
    # step_size: None or (ARGN,)
    # updating by stepSize * (+/-)acceleration and stepSize * (+/-)(1 / acceleration)
    # shrinking when updating failed
    # Return:
    # _arg_cur: (1, ARGN)
    # _I_best: (n=1, c, hp, wp)
    # _loss_min
    # rounds: executed rounds
    def hillClimb(self, _arg, _I, _arg_min, _arg_max, lf, draw, lv = 0, 
                  step_size = None, acceleration = 1.2, min_decline = 0.0001, 
                  gray = False,
                  rounds = 100, _foreground = None):
        
        primitives, dimensions = _arg.shape
        if gray:
            dimensions -= 2
        if step_size is None:
            _step_size = clone(self._step_size)
        else:
            _step_size = self.tensor(step_size) # (ARGN,)

        _min_decline = self.tensor(min_decline)
        _arg_cur = clone(_arg)
        _shrink_rate = self.tensor(acceleration)
        _candidate = self.tensor([-acceleration, -1.0/acceleration, 1.0/acceleration, acceleration]) # (4, )
    
        _min_loss = None
        i_r = 0
        if self.debug:
            self._debugData = self.tensor([])
        while i_r < rounds:
            if not(_min_loss is None):
                _before_loss = clone(_min_loss)
            else:
                _before_loss = None

            # Single stroke optimization
            for i in range(dimensions):   
                if (self.fix_sx and i == 2) or (self.fix_theta and i == 4) or (self.keep_aspect and i == 3):
                    continue    

                _step = _step_size[i] * _candidate # (4, )
                _arg_candidate = _arg_cur.repeat((4, 1)) # (1, ARGN) => (4, ARGN)
                _arg_candidate[:, i] = torch.clip(_arg_candidate[:, i] + _step, _arg_min[i], _arg_max[i])
                if self.keep_aspect and i == 2:
                    _arg_candidate[..., 3] = _arg_candidate[..., 2] * self.stroke_aspect


                #(4, ) = (4, ) + (4, )
                
                i_best, _I_best, _loss_best = self.bestSearch(_arg = _arg_candidate, _I = _I, lv = lv,  
                                                         lf = lf, draw = draw, 
                                                         _foreground = _foreground)
                if i_best < 4 and ((_min_loss is None) or (_loss_best < _min_loss)) :
                    _min_loss = _loss_best
                    _arg_cur = _arg_candidate[i_best].unsqueeze(0) #(1, ARGN)
                    _step_size[i] = _step[i_best] #bestStep                 
                else:  # updating failed then shrinking            
                    _step_size[i] /= _shrink_rate    

            if gray:
                _step_size[7:] = _step_size[6]
                _arg_candidate[:, 7:] = _arg_candidate[:, 6]

            if self.debug:
                self._debugData = torch.cat([self._debugData, _min_loss])
                
            if not (_before_loss is None) and ((_before_loss - _min_loss)  < _min_decline):
                i_r = rounds
            else:
                i_r += 1
        return _arg_cur, _I_best, _min_loss, i_r
        #_arg_cur: (1, ARGN)
        #_I_best:(n=1, c, hp, wp)
    # painter::hillClimb 

    # hillClimbBatch
    # _arg: (primitives, dimensions), 
    # _I: (n=1, c, h, w) 
    # _arg_min, _arg_max: (ARGN,)
    # step_size: None or (ARGN,)
    # updating by stepSize * (+/-)acceleration and stepSize * (+/-)(1 / acceleration)
    # shrinking when updating failed
    # Return:
    # _arg_cur: (primitives, ARGN)
    # _I_best: (primitives, c, h, w)
    # _loss_min (primitives, )
    # rounds: executed rounds
    def hillClimbBatch(self, _arg, _I, _arg_min, _arg_max, lf, draw, lv = 0, 
                  step_size = None, acceleration = 1.2, min_decline = 0.0001, 
                  gray = False, rounds = 100, _foreground = None):
        primitives, dimensions = _arg.shape
        if primitives == 1:
            return self.hillClimb(_arg, _I, _arg_min, _arg_max, lf, draw, lv = lv, 
                  step_size = step_size, acceleration = acceleration, min_decline = min_decline, 
                  gray = gray, rounds = rounds, _foreground = _foreground)

        if gray:
            dimensions -= 2
        if step_size is None:
            _step_size = clone(self._step_size)
        else:
            _step_size = self.tensor(step_size) # (ARGN,)
        _step_size = _step_size.repeat([primitives, 1])    # (primitives, ARGN,)     
           
        _min_decline = self.tensor(min_decline)
        _arg_cur = clone(_arg)
        _shrink_rate = self.tensor(acceleration)
        _candidate = self.tensor([-acceleration, -1.0/acceleration, 1.0/acceleration, acceleration]) # (4, )
    
        _min_loss = self._largest_loss
        i_r = 0
        if self.debug:
            self._debugData = self.tensor([])
        while i_r < rounds:
            _before_loss = _min_loss

            # Stroke batch optimization
            for i in range(dimensions): 
                if (self.fix_sx and i == 2) or (self.fix_theta and i == 4) or (self.keep_aspect and i == 3):
                    continue      

                _step_size_i = _step_size[:, i] # (primitives, )      
                _step = _step_size_i.unsqueeze(-1) * _candidate # (primitives, 1) * (4, ) = (primitives, 4)
                _arg_candidate = _arg_cur.unsqueeze(1).repeat(1, 4, 1) # (primitives, 4, ARGN)
                _arg_candidate[..., i] = torch.clip(_arg_candidate[..., i] + _step, _arg_min[i], _arg_max[i])
                #(primitives, 4) = (primitives, 4) + (primitives, 4)
                if self.keep_aspect and i == 2:
                    _arg_candidate[..., 3] = _arg_candidate[..., 2] * self.stroke_aspect
                    #print('hillClimbBatch::self.stroke_aspect', self.stroke_aspect)
                    #print('hillClimbBatch::self._arg_candidate', _arg_candidate)

                _arg_best, _I_best, _loss_best, _step_best, _i_best = self.bestSearchBatch(
                                                            _arg_set = _arg_candidate, _I = _I, lv = lv,  
                                                            lf = lf, draw = draw, _foreground = _foreground, 
                                                            _step = _step)
                # _arg_best: (primitives, ARGN)
                # _I_best: (primitives, c, h, w)
                # _loss_best: (primitives, )
                # _step_best: (primitives, )
                # _i_best: (primitives, ), The index of the best primtive argument of each batch  

                _chk = _loss_best < _min_loss # (primitives, )            
                _arg_cur = torch.where(_chk.unsqueeze(-1), _arg_best, _arg_cur)
                _min_loss = torch.where(_chk, _loss_best, _min_loss)
                _step_size_i = torch.where(_chk, _step_best, _step_size_i / _shrink_rate)
                _step_size[:, i] = _step_size_i
            if gray:
                _step_size[:, 7:] = _step_size[:, 6]
                _arg_candidate[..., 7:] = _arg_candidate[..., 6]

            if self.debug:
                self._debugData = torch.cat([self._debugData, _min_loss])
                
            if ((_before_loss - _min_loss)  < _min_decline).all():
                i_r = rounds
            else:
                i_r += 1
        return _arg_cur, _I_best, _min_loss, i_r
        #_arg_cur: (1, ARGN)
        #_I_best:(n=1, c, hp, wp)
    # painter::hillClimbBatch    
        

    # Creating an error map
    # _I, destination image in torch.tensor
    # thres_r, the minimum error, an error value >= thres_r will be kept as one; otherwise, zero.
    # sigma, Gaussian filtering's width, no filtering if sigma <= 0.0.
    # meta, True returns metadata.
    # Return:
    # an errmap of (n =1, c=1, H, W)
    def errmap(self, _I = None, thres_r = 0.25, sigma = 1., kernel_size = 3, lv = 0, meta = False):    
        if _I is None:
            _I = self._I[lv]
        _Id = (_I - self._target[lv]).abs().mean(dim = 1, keepdim = True) # (n=1, c=1, h, w)
        #print('_Id.max', _Id.max())
        if sigma > 0.0:
            blurrer = GaussianBlur(kernel_size=kernel_size, sigma=sigma)
            _Id = blurrer(_Id)
        #print('_Id.max', _Id.max())
        _I_map = torch.where(_Id >= thres_r, 1.0, 0.0)

        if meta:
            return _I_map, _Id
        else:
            return _I_map   
    # painter::errmap

    def initGaussianEllipsePCA(self, err_map, center_p, show_fig = False):
        map_h, map_w = err_map.shape
        if map_h < map_w:
            d = map_w - map_h
            dt = d // 2
            err_map_exp = np.zeros([map_w, map_w])
            err_map_exp[dt:dt + map_h, :] = err_map
            err_map = err_map_exp
        elif map_w < map_h:
            d = map_h - map_w
            dt = d // 2
            err_map_exp = np.zeros([map_h, map_h])
            err_map_exp[:, dt:dt + map_w] = err_map
            err_map = err_map_exp
        map_h, map_w = err_map.shape

        axis_y = np.linspace(0.5, -0.5, map_h)
        axis_x = np.linspace(-0.5, 0.5, map_w)
        idx = np.where(err_map > 0)
        Y = axis_y[idx[0]]
        X = axis_x[idx[1]]

        P = np.stack([X, Y], axis = -1 )

        Pt = self.pca.fit_transform(P)
        img_size = np.array([map_w, map_h])
        center_bias = self.pca.mean_    

        g = np.zeros(ARGN)
        g[0:2] = center_p + center_bias * img_size # the center location in Cartersion 
        pt_min = Pt.min(axis = 0)
        pt_max = Pt.max(axis = 0)
        g[2:4] = (pt_max - pt_min) * img_size / 2
        
        g[4] = np.arccos(self.pca.components_[0, 0])
        #print('gi.pca.components_[0, 0]', self.pca.components_[0, 0], ', angle:', g[4])
        if (self.pca.components_[0, 1] >= 0.0):
            g[4] = -g[4]

        if show_fig:
            vgi.showImage(err_map)
            print('X', X.shape)
            print('Y', Y.shape)
            plt.plot(X, Y, 'ro')
            plt.show()      
            print('components:', self.pca.components_)
            print('|components|:', np.linalg.norm(self.pca.components_, axis = 0))
            print('mean_:', self.pca.mean_)
            print('explained_variance_:', self.pca.explained_variance_)        
            plt.plot(Pt[:, 0], Pt[:, 1], 'bo')
            plt.show()
        g = np.expand_dims(g, 0)
        return g  
        
    def initGEPCA(self, err_map, err_data, bb, lv = 0, min_px = 5, div_shape = (64, 64), alpha_ini = 0.9, show_fig = False, save_fig_path = None):
        err_map_bb = err_map[bb]
        err_data_bb = err_data[bb]
        img_cc = self.target_pyd[lv] * np.expand_dims(err_map, -1) 
        map_h, map_w = err_map_bb.shape
        g = None
        errmeans = []
        em_bb = None
        if not(save_fig_path is None):
            em_bb_div = np.array(err_map_bb)

        for row_s in range(0, map_h, div_shape[0] // 4 * 2):
            row_e = row_s + div_shape[0]
            slc_row = slice(row_s, row_e)
            row_s_img = row_s + bb[0].start
            slc_row_img = slice(row_s_img, row_s_img + div_shape[0])
            for col_s in range(0, map_w, div_shape[1] // 4 * 2):
                col_e = col_s + div_shape[1]    
                slc_col = slice(col_s, col_e)
                bb_div = (slc_row, slc_col)

                err_map_div = err_map_bb[bb_div]
                n_px_div = int(np.sum(err_map_div))
                #print('err_map_div', err_map_div.shape, ', n_px_div', n_px_div)
                if n_px_div >= min_px: 
                    if not(save_fig_path is None):
                        #div_bd = np.zeros(err_map_div.shape)
                        #div_bd[0, :] = div_bd[-1, :] = 1.0
                        #div_bd[:, 0] = div_bd[:, -1] = 1.0
                        div_bd = np.ones(err_map_div.shape)
                        em_bg = em_bb_div[bb_div]
                        em_bb_div[bb_div] = 0.5 * div_bd + (1 - 0.5 * div_bd) * em_bg
                    col_s_img = col_s + bb[1].start
                    slc_col_img = slice(col_s_img, col_s_img + div_shape[1] )
                    bb_div_img = (slc_row_img, slc_col_img)                     
                    div_h, div_w = err_map_div.shape                
                    center_r = slc_row_img.start + div_h / 2
                    center_c = slc_col_img.start + div_w / 2
                    #print('slc_col_img', slc_col_img)
                    #print('center_r, center_c', center_r, center_c)
                    center_p = self.ij2xy(np.array([center_r, center_c]), lv = lv)                
                    g_k = self.initGaussianEllipsePCA(err_map_div, center_p, show_fig = show_fig)
                    g_k[0, 5] = alpha_ini                
                    img_div = img_cc[bb_div_img]
                    g_k[0, 6:] = np.sum(img_div, axis = (0, 1)) / n_px_div
                    if g is None:
                        g = g_k
                    else:
                        g = np.concatenate([g, g_k])

                    err_data_div = err_data_bb[bb_div]
                    errmean = err_data_div.sum() / n_px_div
                    errmeans += [errmean]

                # n_px_div >= min_px? 
            # column division loop
        # row division loop
        if not(save_fig_path is None):
            em_bb_div = np.clip(em_bb_div, 0.0, 1.0)
            em_fig = np.tile(np.expand_dims(err_map, -1), [1, 1, 3])
            em_fig_r = em_fig[..., 0]
            em_fig_r[bb] = em_bb_div
            em_fig[..., 0] = em_fig_r
            vgi.saveImage(save_fig_path, em_fig, revChannel = True)
        return g, errmeans       

    # Creating a list of Gaussian ellipses by errmap, CCL, and PCA.
    # the shape of target must be (H, W, C), even a grayscale image, (H, W, 1).
    # err_map, an errmap of (H, W), or a torch.tensor in (1, 1, h, w)
    # err_data, the differences of all pixels, (H, W), or a torch.tensor in (1, 1, h, w) 
    # min_px, the minimun area of connected component.
    # alpha_ini, the initial value of alpha of each GE.
    # sort_loss, sorting by loss in ascending order.
    # meta, True returns the metadata.
    # Return:
    #   a list of GE, (n, 9)
    #i_save_em = 0
    def errorGE(self, _err_map, _err_data, lv = 0, min_px = 5, alpha_ini = 0.9, div_shape = (64, 64), shuffle = True,
                meta = False, save_fig_dir = None):
        err_map = vgi.toNumpy(_err_map.squeeze(0).squeeze(0)) if torch.is_tensor(_err_map) else _err_map
        h, w = err_map.shape
        _err_data = _err_data.reshape([h, w])
        n_img_px = self.pixels[lv]
        err_cc = cc3d.connected_components(err_map)  
        #print('err_cc', err_cc.shape, vgi.metric(err_cc))
        #vgi.showImage(err_cc)
        stats = cc3d.statistics(err_cc)
        #print(stats.keys())
        voxel_counts = stats['voxel_counts']
        bounding_boxes = stats['bounding_boxes']
        centroids = stats['centroids']
        n = voxel_counts.size

        strokes = []
        err_stk = np.array(err_map)
        fig_div_path = None
        #err_stk = np.expand_dims(err_stk)
        for i in range(1, n):
            bb = bounding_boxes[i]  
            bb2d = (bb[0], bb[1])
            err_map_cc = np.where(err_cc == i, 1., 0.) # whole image with label == i
   
            #print('bb', bb)
            #print('err_map_i', i)
            #vgi.showImage(err_map_i)

            if voxel_counts[i] >= min_px:

                if not(save_fig_dir is None):
                    fig_div_path = save_fig_dir + str(i) + '_div.png'
                #gb = self.initGeoGEGMM(err_map_cc, bb2d, div_area = div_area, alpha_ini = alpha_ini, min_px = min_px)
                gb, errmeans = self.initGEPCA(err_map_cc, _err_data, bb2d, 
                                              lv = lv, div_shape = div_shape, alpha_ini = alpha_ini, min_px = min_px,
                                              save_fig_path = fig_div_path)
                if not (gb is None):
                    for k, g_k in enumerate(gb):
                        g_k_size = g_k[2] * 2 * g_k[3] * 2
                        if g_k_size < min_px:
                            continue
                        stroke = {'size':g_k_size, 'g':np.expand_dims(g_k, 0)}
                        strokes += [stroke]

                    if not(save_fig_dir is None): # Draw each CCL with strokes
                        fig_alpha = 0.5 
                        img_fig = np.tile(np.expand_dims(err_map, -1), [1, 1, 3])                        
                        print('|gb|:', gb.shape[0])

                        g_fig = np.array(gb)
                        g_fig[..., 5] = fig_alpha # alpha
                        g_fig[..., 6:] = np.array([1.0, 0.0, 0.0]) # color

                        _arg_fig = self.validarg(g_fig)
                        _I_fig = self.composite(_arg_fig, lv = lv, primitive = 'rect', update_I = False, no_prev = True)
                        I_fig = toNumpyImage(_I_fig)
                        I_alpha = I_fig[..., 0]
                        #I_fig = I_alpha * I_fig + (1-I_alpha) * np.expand_dims(err_map_cc, -1)

                        I_fig[..., 0] = I_fig[..., 0] + (1-I_alpha) * err_map_cc
                        I_fig[..., 1] = err_map_cc
                        I_fig[..., 2] = err_map_cc
                        vgi.showImage(I_fig)

                        fig_cc_name = str(i) + '_cc.png'
                        fig_cc_path = save_fig_dir + fig_cc_name
                        print('fig_cc_path', fig_cc_path)
                        vgi.saveImage(fig_cc_path, err_map_cc, revChannel = True)                        

                        fig_name = str(i) + '.png'
                        fig_path = save_fig_dir + fig_name
                        print('fig_path', fig_path)
                        vgi.saveImage(fig_path, I_fig, revChannel = True)


            else:
                err_stk[bb2d] = err_stk[bb2d] * (1 - err_map_cc[bb2d])
        if shuffle:
            random.shuffle(strokes)

        strokes.sort(key = itemgetter('size'), reverse = True)
        g = None
        for k, stroke in enumerate(strokes):
            g_k = stroke['g'] 

            # debug
            #if i == 0:
            #    vgi.saveImage('em_' + str(self.i_save_em) + '.png', stroke['em'])
            #    self.i_save_em += 1
            if g is None:
                g = g_k
            else:               
                g = np.concatenate([g, g_k])
              
        if meta:
            return g, strokes, err_stk    
        else:
            return g  
    # painter::errorGE    


    # Deciding the shape of a sub-region for stroke initializtion.
    # n_max: the maximum stroke number
    # lv_start: starting level of image pyramid, default -1.
    # return: (height, width)
    def divShape(self, n_max, lv_start = -1):
        _, _, h_lv, w_lv = self.shape[lv_start]
        log_shape_lv = np.ceil(np.log2([h_lv, w_lv]))
        if n_max <= 100:
            log_shape_lv -= 1
        elif n_max <= 250:
            log_shape_lv -= 2
        else:
            log_shape_lv -= 3
        log_shape_lv = np.clip(log_shape_lv, 0, None)
        div_shape = (2 ** log_shape_lv).astype(np.int32)
        return div_shape
    # painter::divShape

    # ---------------------------------------------------------------
    def paint( self, n_max = 3000, loss = 'maemse', init_type = 'em', reopt = 0,
               primitive = 'brush', brush_path = None, brush_inv = False, brush_masking = True,                
               lv_start = -1, lv_target = 0, lv_rep = 1, 
               rounds = 100, min_decline = 0.000001, rounds_reopt = None, min_decline_reopt = None,   
               thres_r = 0.3, min_thres_r = 0.025, CCL_sigma = 1.0, min_px = 3, alpha_ini = 0.8, div_shape = None,
               thres_r_w = 0.8, CCL_sigma_w = 0.9, alpha_ini_w = 1.0, shuffle = True, 
               verbose = 1):

        acc = 1.2

        img_shape = self.target.shape
        height, width, channels = img_shape

        self.brush_masking = brush_masking
        if not brush_path is None:
            self.loadBrush(brush_path, brush_inv)
            primitive = 'brush0'

        if lv_start < 0:
            lv_start = self.pyd_levels - lv_start
        lv_start = np.clip(lv_start, 0, self.pyd_levels - 1)

        if div_shape is None:
            div_shape = self.divShape(n_max, lv_start)

        if verbose & 1:
            timeS = time.time()
            print('pyd_levels:',  self.pyd_levels)
            print('lv_start:',  lv_start)
            print('div_shape:', div_shape)
            print('background:', self.bg)     

        lf = self.getLoss(loss) 
        draw = self.getDrawMethod(primitive)
        opt = self.hillClimbBatch
        _arg_min = self.tensor(self.arg_min[lv_start])
        _arg_max = self.tensor(self.arg_max[lv_start])    
        n = 0
        save_err_fig = verbose & 1024 
        save_fig_dir = None
        save_fig_dir0, save_fig_i = 'fig_stroke_init/', 0 if save_err_fig else None

        while n < n_max:        
            _I = self._I[lv_start]

            arg_ini = None
            if init_type == 'em':
                _em, _Id = self.errmap(_I = _I, lv = lv_start, thres_r = thres_r, sigma = CCL_sigma, meta = True)
                if save_err_fig:
                    save_fig_dir = save_fig_dir0 + str(save_fig_i) + '/'
                    if not os.path.exists(save_fig_dir):
                        os.makedirs(save_fig_dir)
                arg_ini = self.errorGE(_em, _Id, lv = lv_start, min_px = min_px, alpha_ini = alpha_ini, div_shape = div_shape, shuffle = shuffle, save_fig_dir = save_fig_dir)        
                
            elif init_type == 'random':
                arg_ini = self.randomInitArg(n_max, lv = lv_start, factor_u = 0.5, factor_s = 0.5, min_size = min_size, shrink_rate = 0.9, shrink_min = 0.05, shrink_n = 10)  


            n_ini = 0 if arg_ini is None else arg_ini.shape[0]
            if verbose & 4:
                print(init_type, 'arg_ini:', n_ini, ', thres_r:', thres_r, ', CCL_sigma:', CCL_sigma, ', alpha_ini:', alpha_ini)

            thres_r *= thres_r_w
            thres_r = max(thres_r, min_thres_r)
            CCL_sigma *= CCL_sigma_w
            alpha_ini *= alpha_ini_w

            if n_ini == 0:
                continue

            if self.keep_aspect:
                arg_ini[..., 3] = arg_ini[..., 2] * self.stroke_aspect


            _arg_ini = self.validarg(arg_ini) #(n_ini, ARGN)
            if verbose & 2:
                _I_ini = self.composite(_arg_ini, lv = lv_start, primitive = primitive, update_I = False, no_prev = True)
                I_ini = toNumpyImage(_I_ini)
                _I_ini_bg = self.composite(_arg_ini, lv = lv_start, primitive = primitive, update_I = False, no_prev = False)
                I_ini_bg = toNumpyImage(_I_ini_bg)            
            i = 0            
            j = 0
            n_d = min(n_max - n, n_ini)
            while i < n_d:
                # batch selection
                j = np.clip(i + self.search_batch_size, None, n_d)
                n_b = j - i
                _arg_ini_i = _arg_ini[i:i+1] # at least one stroke in each batch
                if n_b > 1:
                    for s in range(i+1, j):
                        _arg_s = _arg_ini[s]
                        _rs = _arg_s[2:4].max() * self.area_f
                        _rt = _arg_ini_i[:, 2:4].max(dim = -1)[0] * self.area_f # max(dim) returns (max, idx) 
                        if circlex(_arg_s[0], _arg_s[1], _rs, _arg_ini_i[:, 0], _arg_ini_i[:, 1], _rt):
                            break
                        else:
                            _arg_ini_i = torch.cat([_arg_ini_i, _arg_s.unsqueeze(0)])
                n_b = _arg_ini_i.shape[0]
                # n_b > 1?
                i += n_b

                _I_prev = self._I[lv_start]
                if rounds == 0:
                    _arg_cur = _arg_ini_i
                    _loss_best = None
                else:
                    _arg_cur, _I_best, _loss_best, opt_rounds = opt(_arg_ini_i, _I_prev, lv = lv_start, lf = lf, draw = draw,
                                                 _arg_min = _arg_min, _arg_max = _arg_max, acceleration = acc, 
                                                 gray = False, min_decline = min_decline, rounds = rounds)
                # _arg_cur: (n_b, ARGN)
                # _I_best: (n_b, c, h, w)
                # _loss_best: (n_b,)

                # Updating the result image 
                _I_cur = self.composite(_arg_cur, primitive = primitive, lv = lv_start, batch_size = n_b)

                self.appendArg(_arg_cur, lv = lv_start)

                if self.debug:
                    Err = toNumpy(self._debugData)
                    Rounds = range(Err.size)
                    plt.plot(Rounds, Err, 'go--', linewidth=1, markersize=3)
                    plt.show()
                n += n_b
                if n >= n_max:
                    break

            if verbose & 1:
                t = time.time() - timeS
                print('n:', n, ', time:', '%0.8f'%t)

            if verbose & 2:
                I = self.image(lv_start)
                em = toNumpyImage(_em)
                em = np.where(em > 0.0, 1.0, 0.0)
                vgi.showImageTable([em, I_ini, I_ini_bg, I], 1, 4, figsize = (12, 5))       
            if save_err_fig:
                print('em', vgi.metric(em))
                Id = vgi.normalize(toNumpyImage(_Id))
                vgi.saveImage(save_fig_dir + 'Id.png', Id, revChannel = True)
                vgi.saveImage(save_fig_dir + 'em.png', em, revChannel = True)
                vgi.saveImage(save_fig_dir + 'I_ini.png', I_ini, revChannel = True)
                vgi.saveImage(save_fig_dir + 'I_ini_bg.png', I_ini_bg, revChannel = True)
                vgi.saveImage(save_fig_dir + 'I.png', I, revChannel = True)
                save_fig_i += 1

            # _arg_ini loop
        # round loop
        lv = lv_start
        lv_rep_c = lv_rep
        rounds_reopt = rounds if rounds_reopt is None else rounds_reopt
        min_decline_reopt = min_decline if min_decline_reopt is None else min_decline_reopt

        if verbose & 1:
            print('Backward optimization (BO)', reopt, ', lv:', lv, '>>', lv_target, ', lv_rep:', lv_rep)   

        for i in range(reopt):
            # upscale       
            if lv > lv_target:
                if lv_rep_c == lv_rep:
                    self.upscaleArg(lv) 
                    lv -= 1
                    lv_rep_c = 1
                    self.composite(lv = lv, primitive = primitive, update_I = True)
                    if verbose & 2:
                        I = self.image(lv)
                        print('upscale, lv:', lv)
                        vgi.showImage(I)
                else:
                    lv_rep_c += 1
                       
            self.bwopt(lv = lv, loss = loss, primitive = primitive,
                       rounds = rounds_reopt, min_decline = min_decline_reopt, batch_size = self.search_batch_size,
                       verbose = verbose)
            if verbose & 1:
                t = time.time() - timeS
                print('BO', i, 'level:', lv, ', time:', '%0.8f'%t)      
        
        while lv > lv_target: 
            self.upscaleArg(lv) 
            lv -= 1
            self.composite(lv = lv, primitive = primitive, update_I = True)
        return  self.image(lv)
    # painter::paint


    # backward optimization
    def bwopt(self, lv = 0, loss = 'maemse', primitive = 'brush0',
              rounds = 100, min_decline = 0.000001, batch_size = 8, n = 0, clamp = True,
              verbose = 1):

        shape_I = self.shape[lv]
        images, channels, height, width = shape_I
        primitives, dimensions = self._A[lv].shape
        pixels = self.pixels[lv] 

        bs_count = [0] * batch_size


        if n <= 0:
            n = primitives

        if verbose & 1:
            timeS = time.time()
            #print('reoptimize #primitives:', primitives)     

        lf = self.getLoss(loss) 
        draw = self.getDrawMethod(primitive)
        opt = self.hillClimbBatch
        acc = 1.2

        _arg_min = self.tensor(self.arg_min[lv])
        _arg_max = self.tensor(self.arg_max[lv])

        _arg = clone(self._A[lv])
        _I_bg = clone(self._I[lv])
        _I_fg = self.zeros((images, channels, height, width)) # (n, c, h, w)
        _f_kp = self.ones( (images, 1, height, width) ) # (n, 1, h, w)

        if verbose & 2:
            print('Initial image:')
            vgi.showImage(self.image(lv))
            #print('reoptimize #primitives:', primitives)     


        k = primitives #[ks:k]
        n_prim = 0
        area_f = self.area_f * 2
        while k > 0:
            # batch selection (reopt)
            ks = np.clip(k - batch_size, 0, None)
            n_b = k - ks
            _arg_b = _arg[k-1:k] # at least one stroke in each batch
            if n_b > 1:
                for s in range(k-2, ks-1, -1):
                    _arg_s = _arg[s]
                    _rs = _arg_s[2:4].max() * self.area_f
                    _rt = _arg_b[:, 2:4].max(dim = -1)[0] * self.area_f # max(dim) returns (max, idx) 
                    if circlex(_arg_s[0], _arg_s[1], _rs, _arg_b[:, 0], _arg_b[:, 1], _rt):
                        break
                    else:
                        _arg_b = torch.cat([_arg_s.unsqueeze(0), _arg_b])
            # n_b > 1?
            n_b = _arg_b.shape[0]
            ks = k - n_b
            bs_count[n_b - 1] += 1
            _I_Ga, _f, _G = draw(_arg_b, lv = lv)
                #  _I_Ga: (primitives, channels, h, w) if is_unisize; otherwise, [primitives, (channels, patch_height, patch_width)]; the images of all Gaussians, alpha * G
                #  _f: (primitives, h, w), 1 - alpha * Q    
                #  _G: (primitives, h, w)          
            _foreground = (_I_fg, _f_kp)

            # remove the strokes
            # Backward alpha compositing: I_k  = \beta_k \alpha_k G_k + (1 - \alpha_k G_k)I_{k-1}
            #   Inverse:  I_{k-1} = (I_k - \beta_k \alpha_k G_k) / (1 - \alpha_k G_k)
            i = n_b - 1
            while i >= 0:
                _I_Ga_i = _I_Ga[i] # (c, h, w)
                _f_i = _f[i]       # (1, h, w)
                _I_bg = (_I_bg - _I_Ga_i) / _f_i
                if clamp:
                    torch.clamp(_I_bg, min=0.0, max=1.0, out=_I_bg)
                i -= 1

            # Updating
            ts_opt = time.time()
            _arg_k, _I_best, _loss_best, opt_rounds = opt(_arg_b, _I_bg, lv = lv, lf = lf, draw = draw,
                                         _arg_min = _arg_min, _arg_max = _arg_max,
                                         acceleration = acc, min_decline = min_decline, rounds = rounds,
                                         gray = False, _foreground = _foreground)  
            
            if verbose & 4:
                t_opt = time.time() - ts_opt
                print('opt time:', '%.5f'%t_opt, ', bs:', n_b, ', rounds:', opt_rounds, 'avg time:', '%.6f'%(t_opt/rounds/n_b))
            # Updating the result image
            #_I_k = self.composite(_arg_k, lv = lv, batch_size = n_b)
            _arg[ks:k] = _arg_k

            _I_Ga_k, _f_k, _G_k= draw(_arg_k, lv = lv)

            _I_k = clone(_I_bg)
            #for i in range(n_b-1, -1, -1):
            #    _I_k =  _I_Ga_k[i] + _f_k[i] * _I_k
            for i in range(n_b):
                _I_k =  _I_Ga_k[i] + _f_k[i] * _I_k

            _I_k = _I_fg + _f_kp * _I_k
            self._I[lv] = _I_k

            for i in range(n_b-1, -1, -1):
                _I_fg += _I_Ga_k[i] * _f_kp
                _f_kp *= _f_k[i]      

            n_prim += n_b

            if verbose & 2:
                print('n_prim:', n_prim)
                I = self.image(lv = lv)            
                vgi.showImage(I)      

            # batch OK
            k -= n_b     
            if n_prim >= n:
                break;         
        # primitive loop
        if verbose & 4:
            print('batch_size count:', bs_count)

        self._A[lv] = clone(_arg)
        return self._A[lv] 
    # painter::bwopt
# @painter
# ---------------------------------------------------------
def accuracyConfig(accuracy):
    if accuracy == 0:
        min_decline, reopt = 9999999, 0
    elif accuracy == 1:
        min_decline, reopt = 0.00001, 0
    elif accuracy == 2:
        min_decline, reopt = 0.00001, 1
    elif accuracy == 3:
        min_decline, reopt = 0.000001, 1
    elif accuracy == 4:
        min_decline, reopt = 0.000001, 3
    elif accuracy == 5:
        min_decline, reopt = 0.000001, 5
    elif accuracy == 6:
        min_decline, reopt = 0.000001, 7
    elif accuracy == 7:
        min_decline, reopt = 0.000001, 10
    else:
        min_decline, reopt = 0.000001, 3
    return min_decline, reopt
       
# ---------------------------------------------------------
def repaint(arg, shape, bg = [0.0, 0.0, 0.0], max_n = -1, 
           batch = 10, save_batch = False, scale_x = None, scale_y = None, 
           primitive = 'brush0', brush_path = None, brush_inv = False, brush_masking = True,
           weight_only = False, gamma = None, alpha = None, 
           tensor = False, pt = None):
    img_shape = list(shape) # (H, W, 3)
    if not(scale_x is None):
        img_shape[1] = int(img_shape[1] * scale_x)        
        arg[:, 0] *= scale_x
        arg[:, 2] *= scale_x
    if not(scale_y is None):                
        img_shape[0] = int(img_shape[0] * scale_y)
        arg[:, 1] *= scale_y
        arg[:, 3] *= scale_y       

    if not(alpha is None):
        arg[:, 5] = alpha

    iB = 0
    iQ = 0
    if pt is None:
        pt = painter(target = np.zeros(img_shape), bg = bg,)
    else:
        pt.setBackground(bg)

    pt.brush_masking  = brush_masking
    if not brush_path is None:
        pt.loadBrush(brush_path, brush_inv)
        primitive = 'brush0'

    n = arg.shape[0]
    if save_batch:
        imgSet = []

    if max_n >= 0:
        n = min(arg.shape[0], max_n)
    while iQ < n:
        iQe = min(iQ + batch, n)
        argB = arg[iQ:iQe, :]
        pt.setArg(argB)
        if weight_only:
            pt.compositeWeightOnly(primitive = primitive)
        else:
            pt.composite(primitive = primitive)
        if not(gamma is None):
            pt.gamma = gamma
        if save_batch:
            I =  pt._I if tensor else pt.image()
            imgSet += [I]
        iB += 1
        iQ = iQe
    pt.setArg(arg[:n])
    I = None
    if save_batch:
        I = imgSet
    else:
        I = pt._I if tensor else pt.image()

    return I
# repaint  
