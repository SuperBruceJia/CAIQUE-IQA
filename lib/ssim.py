# -*- coding: utf-8 -*-

import torch
import torch.jit
import torch.nn.functional as F


@torch.jit.script
def create_window(window_size: int, sigma: float, channel: int):
    '''
    Create 1-D gauss kernel
    :param window_size: the size of gauss kernel
    :param sigma: sigma of normal distribution
    :param channel: input channel
    :return: 1D kernel
    '''
    coords = torch.arange(window_size, dtype=torch.float)
    coords -= window_size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    g = g.reshape(1, 1, 1, -1).repeat(channel, 1, 1, 1)
    return g


@torch.jit.script
def _gaussian_filter(x, window_1d, use_padding: bool):
    '''
    Blur input with 1-D kernel
    :param x: batch of tensors to be blured
    :param window_1d: 1-D gauss kernel
    :param use_padding: padding image before conv
    :return: blured tensors
    '''
    C = x.shape[1]
    padding = 0
    if use_padding:
        window_size = window_1d.shape[3]
        padding = window_size // 2
    out = F.conv2d(x, window_1d, stride=1, padding=(0, padding), groups=C)
    out = F.conv2d(out, window_1d.transpose(2, 3), stride=1, padding=(padding, 0), groups=C)
    return out


@torch.jit.script
def ssim(X, Y, window, data_range: float, use_padding: bool=False):
    '''
    Calculate ssim index for X and Y
    :param X: images
    :param Y: images
    :param window: 1-D gauss kernel
    :param data_range: value range of input images. (usually 1.0 or 255)
    :param use_padding: padding image before conv
    :return:
    '''

    K1 = 0.01
    K2 = 0.03
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    mu1 = _gaussian_filter(X, window, use_padding)
    mu2 = _gaussian_filter(Y, window, use_padding)
    sigma1_sq = _gaussian_filter(X * X, window, use_padding)
    sigma2_sq = _gaussian_filter(Y * Y, window, use_padding)
    sigma12 = _gaussian_filter(X * Y, window, use_padding)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (sigma1_sq - mu1_sq)
    sigma2_sq = compensation * (sigma2_sq - mu2_sq)
    sigma12 = compensation * (sigma12 - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    # Fixed the issue that the negative value of cs_map caused ms_ssim to output Nan.
    cs_map = F.relu(cs_map)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_val = ssim_map.mean(dim=(1, 2, 3))  # reduce along CHW
    cs = cs_map.mean(dim=(1, 2, 3))

    return ssim_val, cs


class SSIM(torch.jit.ScriptModule):
    __constants__ = ['data_range', 'use_padding']

    def __init__(self, window_size=11, window_sigma=1.5, data_range=255., channel=3, use_padding=False):
        '''
        :param window_size: the size of gauss kernel
        :param window_sigma: sigma of normal distribution
        :param data_range: value range of input images. (usually 1.0 or 255)
        :param channel: input channels (default: 3)
        :param use_padding: padding image before conv
        '''
        super().__init__()
        assert window_size % 2 == 1, 'Window size must be odd.'

        window = create_window(window_size, window_sigma, channel)
        self.register_buffer('window', window)
        self.data_range = data_range
        self.use_padding = use_padding

    @torch.jit.script_method
    def forward(self, X, Y):
        r = ssim(X, Y, window=self.window, data_range=self.data_range, use_padding=self.use_padding)
        return r[0]


# import torch
# import torch.nn.functional as F
# from torch.autograd import Variable
# import torch.jit
#
# import numpy as np
# from math import exp
#
#
# def gaussian(window_size, sigma):
#     gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
#
#     return gauss / gauss.sum()
#
#
# def create_window(window_size, channel):
#     _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
#     _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
#     window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
#
#     return window
#
#
# @torch.jit.script
# def _ssim(img1, img2, window, window_size, channel, size_average=True):
#     mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
#     mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
#
#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2)
#     mu1_mu2 = mu1 * mu2
#
#     sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
#     sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
#     sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
#
#     C1 = 0.01 ** 2
#     C2 = 0.03 ** 2
#
#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
#
#     if size_average:
#         return ssim_map.mean(1).mean(1).mean(1)
#     else:
#         return ssim_map.mean(1).mean(1).mean(1)
#
#
# class SSIM(torch.nn.Module):
#     def __init__(self, window_size=11, size_average=True):
#         super(SSIM, self).__init__()
#         self.window_size = window_size
#         self.size_average = size_average
#         self.channel = 1
#         self.window = create_window(window_size, self.channel)
#
#     @torch.jit.script
#     def forward(self, img1, img2):
#         (_, channel, _, _) = img1.size()
#
#         if channel == self.channel and self.window.data.type() == img1.data.type():
#             window = self.window
#         else:
#             window = create_window(self.window_size, channel)
#
#             if img1.is_cuda:
#                 window = window.cuda(img1.get_device())
#             window = window.type_as(img1)
#
#             self.window = window
#             self.channel = channel
#
#         return _ssim(img1, img2, window, self.window_size, channel, self.size_average)
#
#
# @torch.jit.script
# def ssim(img1, img2, window_size=11, size_average=True):
#     (_, channel, _, _) = img1.size()
#     window = create_window(window_size, channel)
#
#     if img1.is_cuda:
#         window = window.cuda(img1.get_device())
#     window = window.type_as(img1)
#
#     return _ssim(img1, img2, window, window_size, channel, size_average)
