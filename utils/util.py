import os
import math
from scipy.stats import pearsonr
from datetime import datetime
import numpy as np
from PIL import Image
import cv2
from .pan_metrics import q2n, HQNR, interp23
from .metrics import SAM, ERGAS

import transplant
# matlab = transplant.Matlab(executable='/usr/local/MATLAB/R2018a/bin/matlab')
# matlab.addpath('utils/Quality_Indices')

####################
# miscellaneous
####################

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('[Warning] Path [%s] already exists. Rename it to [%s]' % (path, new_name))
        os.rename(path, new_name)
    os.makedirs(path)


####################
# image convert
####################
def pan_Tensor2np(tensor_list, run_range, img_range):

    def _Tensor2numpy(tensor, run_range):
        array = np.transpose(quantize(tensor, run_range, img_range).numpy(), (1, 2, 0)).astype(np.uint16)
        return array

    return [_Tensor2numpy(tensor, run_range) for tensor in tensor_list]


def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                                [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def ycbcr2rgb(img):
    '''same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                           [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def save_img_np(img_np, img_path, mode='RGB'):
    if img_np.ndim == 2:
        mode = 'L'
    img_pil = Image.fromarray(img_np, mode=mode)
    img_pil.save(img_path)


def quantize(img, rgb_range, img_range):
    pixel_range = img_range / rgb_range
    # return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)
    return img.mul(pixel_range).clamp(0, int(img_range)).round()


###Pan-sharpening#####
def pan_calc_metrics_all(databatch, scale, img_range, FR=False):
    PS = databatch['SR'].astype(np.double)
    
    sensor = databatch.pop('MTF') if databatch.get('MTF') is not None else databatch['SENSOR']

    if not FR:
        # GT = np.array(GT).astype(np.double)/img_range
        # PS = np.array(PS).astype(np.double)/img_range
        # RMSE = (GT - PS)/img_range
        # RMSE = np.sqrt((RMSE*RMSE).mean())
        # cc = CC(GT,PS)
        GT = databatch['HR'].astype(np.double)/img_range
        PS = PS/img_range
        sam = SAM(GT, PS)
        # ergas = round(ERGAS2(PS, GT, scale=scale), 4)
        ergas = ERGAS(GT, PS, scale=scale)
        
        # psnr = mPSNR(GT, PS)
        # Qave = Q_AVE(GT, PS)
        # scc = sCC(GT, PS)
        q2n_value, _ = q2n.q2n(GT,PS, 32, 32)
        # return {'PSNR': psnr, 'SAM':sam, 'ERGAS':ergas, 'Q2n':q2n_value}
        rlt = {'SAM':sam, 'ERGAS':ergas, 'Q2n':q2n_value}
    else:
        I_MS_LR = databatch['LR'].astype(np.double)
        I_MS = interp23.interp23(I_MS_LR, scale).astype(np.double)
        I_PAN = databatch['REF'].astype(np.double)
        HQNR_index, D_lambda, D_S = HQNR.HQNR(PS,I_MS_LR,I_MS,I_PAN, 32, sensor,scale)
        rlt = {'D_lambda':D_lambda, 'D_S':D_S, 'HQNR':HQNR_index}

    return rlt


####################
# metric
####################
def calc_metrics_(img1, img2, crop_border, test_Y=True):
    #
    img1 = img1 / 255.
    img2 = img2 / 255.

    if test_Y and img1.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
        im1_in = rgb2ycbcr(img1)
        im2_in = rgb2ycbcr(img2)
    else:
        im1_in = img1
        im2_in = img2
    height, width = img1.shape[:2]
    if im1_in.ndim == 3:
        cropped_im1 = im1_in[crop_border:height-crop_border, crop_border:width-crop_border, :]
        cropped_im2 = im2_in[crop_border:height-crop_border, crop_border:width-crop_border, :]
    elif im1_in.ndim == 2:
        cropped_im1 = im1_in[crop_border:height-crop_border, crop_border:width-crop_border]
        cropped_im2 = im2_in[crop_border:height-crop_border, crop_border:width-crop_border]
    else:
        raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im1_in.ndim))

    psnr = calc_psnr(cropped_im1 * 255, cropped_im2 * 255)
    ssim = calc_ssim(cropped_im1 * 255, cropped_im2 * 255)
    return psnr, ssim


def calc_psnr(img1, img2):
    # img1 and img2 have range [0, 255]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calc_ssim(img1, img2):

    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')