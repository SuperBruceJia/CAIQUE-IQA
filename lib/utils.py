# -*- coding: utf-8 -*-

import os
from sys import stdout

import shutil
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import misc
from scipy.ndimage.filters import convolve
from scipy import stats

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torch.utils.data
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Function

from lib.ssim import SSIM


def image_show(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def sum_multiply(x):
    num = 0
    for i in range(1, x+1):
        num += i ** 2

    return num


def PLCC_loss(x, y):
    mean_x = x.mean().reshape([-1])
    mean_y = y.mean().reshape([-1])

    x_norm = torch.nn.functional.normalize((x - mean_x).reshape([1, -1]))
    y_norm = torch.nn.functional.normalize((y - mean_y).reshape([1, -1]))
    plcc = torch.nn.functional.cosine_similarity(x_norm, y_norm)

    loss = 1 - plcc

    return loss


def custom_tanh(x, k, m):
    x = (torch.exp((x-m)*k) - torch.exp(-(x-m)*k)) / (torch.exp((x-m)*k) + torch.exp(-(x-m)*k))

    return x


def patch_similarity(args, patch, all_patches):
    """
    :param patch: [1, 3, patch_size, patch_size]
    :param all_patches: [num_patch, 3, patch_size, patch_size]
    :return similar_patch: [K, 3, patch_size, patch_size]
    """
    if patch.dim() != 4:
        patch = patch.unsqueeze(dim=0)

    # For training
    if patch.shape[0] == 1:
        num_batch = patch.shape[0]
        num_patch = all_patches.shape[0] // num_batch

        ssim = SSIM(data_range=255., channel=all_patches.shape[1], window_size=args.window_size)
        similarity = ssim(patch.repeat(num_patch, 1, 1, 1), all_patches)

        # Remove the random patch itself
        similarity = similarity[similarity != torch.max(similarity)]
        similar_all_patches = all_patches[torch.topk(similarity, args.num_internal).indices]

    # For Inference
    else:
        num_patch = all_patches.shape[0]
        num_non_overlapping = patch.shape[0]
        similar_all_patches = ()

        ssim = SSIM(data_range=255., channel=all_patches.shape[1], window_size=args.window_size)
        patch = patch.unsqueeze(dim=1).repeat(1, num_patch, 1, 1, 1).reshape(-1, 3,
                                                                             args.internal_patch_size,
                                                                             args.internal_patch_size)
        pad_all_patches = all_patches.unsqueeze(dim=0).repeat(num_non_overlapping,
                                                              1, 1, 1, 1).reshape(-1, 3,
                                                                                  args.internal_patch_size,
                                                                                  args.internal_patch_size)
        similarity = ssim(patch, pad_all_patches)
        similarity = similarity.reshape(num_non_overlapping, num_patch)

        for i in range(num_non_overlapping):
            one_similarity = similarity[i, :]
            one_similarity = one_similarity[one_similarity != torch.max(one_similarity)]

            similar_patch = all_patches[torch.topk(one_similarity, args.num_internal).indices]
            similar_all_patches += (similar_patch,)

    return similar_all_patches


def trucated_gaussian(x, mean=0., std=1., min=0., max=1.):
    gauss = torch.exp((-(x - mean) ** 2)/(2 * std ** 2))

    return torch.clamp(gauss, min=min, max=max)


# def patch_similarity(args, patch, all_patches):
#     """
#     :param patch: [1, 3, patch_size, patch_size]
#     :param all_patches: [num_patch, 3, patch_size, patch_size]
#     :return similar_patch: [K, 3, patch_size, patch_size]
#     """
#     if patch.dim() != 4:
#         patch = patch.unsqueeze(dim=0)
#
#     num_batch = patch.shape[0]
#     num_patch = all_patches.shape[0] // num_batch
#     # print(patch.shape, all_patches[0].shape)
#
#     # Get the SSIM Similarity
#     # similarity = torch.stack([ssim(img1=patch, img2=all_patches[i].unsqueeze(dim=0), window_size=args.window_size)
#     #                           for i in range(num_patch)])
#     # print(patch.shape, all_patches.shape)
#
#     # For training
#     if patch.shape[0] == 1:
#         ssim = SSIM(data_range=255., channel=all_patches.shape[1])
#         similarity = ssim(patch.repeat(num_patch, 1, 1, 1), all_patches)
#
#         # similarity = ssim(img1=patch.repeat(num_patch, 1, 1, 1), img2=all_patches)
#         # print(patch.shape, all_patches.shape, len(similarity))
#
#         # Remove the random patch itself
#         # similarity = similarity[similarity != torch.max(similarity)]
#         similar_patch = all_patches[torch.topk(similarity, args.num_internal).indices]
#
#     # For Inference
#     else:
#         num_non_overlapping = patch.shape[0]
#         ssim = SSIM(data_range=255., channel=all_patches.shape[1])
#         similarity = ssim(patch.unsqueeze(dim=1).repeat(1, num_patch, 1, 1, 1).reshape(-1, 3, args.internal_patch_size, args.internal_patch_size),
#                           all_patches.unsqueeze(dim=0).repeat(num_non_overlapping, 1, 1, 1).reshape(-1, 3, args.internal_patch_size, args.internal_patch_size))
#
#     # print(len(similarity), torch.topk(similarity, args.num_internal))
#     # similar_patch = all_patches[torch.argmax(similarity)]
#     # print(torch.topk(similarity, args.num_internal))
#     # similar_patch = all_patches[torch.topk(similarity, args.num_internal).indices]
#
#     return similar_patch


class TensorAxis:
    N = 0
    H = 1
    W = 2
    C = 3


class CSFlow:
    def __init__(self, sigma=float(0.1), b=float(1.0)):
        self.b = b
        self.sigma = sigma

    def __calculate_CS(self, scaled_distances, axis_for_normalization=TensorAxis.C):
        self.scaled_distances = scaled_distances
        self.cs_weights_before_normalization = torch.exp((self.b - scaled_distances) / self.sigma)
        # self.cs_weights_before_normalization = 1 / (1 + scaled_distances)
        # self.cs_NHWC = CSFlow.sum_normalize(self.cs_weights_before_normalization, axis_for_normalization)
        self.cs_NHWC = self.cs_weights_before_normalization

    # def reversed_direction_CS(self):
    #     cs_flow_opposite = CSFlow(self.sigma, self.b)
    #     cs_flow_opposite.raw_distances = self.raw_distances
    #     work_axis = [TensorAxis.H, TensorAxis.W]
    #     relative_dist = cs_flow_opposite.calc_relative_distances(axis=work_axis)
    #     cs_flow_opposite.__calculate_CS(relative_dist, work_axis)
    #     return cs_flow_opposite

    # --
    @staticmethod
    def create_using_L2(I_features, T_features, sigma=float(0.5), b=float(1.0)):
        cs_flow = CSFlow(sigma, b)
        sT = T_features.shape
        sI = I_features.shape

        Ivecs = torch.reshape(I_features, (sI[0], -1, sI[3]))
        Tvecs = torch.reshape(T_features, (sI[0], -1, sT[3]))
        r_Ts = torch.sum(Tvecs * Tvecs, 2)
        r_Is = torch.sum(Ivecs * Ivecs, 2)
        raw_distances_list = []
        for i in range(sT[0]):
            Ivec, Tvec, r_T, r_I = Ivecs[i], Tvecs[i], r_Ts[i], r_Is[i]
            A = Tvec @ torch.transpose(Ivec, 0, 1)  # (matrix multiplication)
            cs_flow.A = A
            # A = tf.matmul(Tvec, tf.transpose(Ivec))
            r_T = torch.reshape(r_T, [-1, 1])  # turn to column vector
            dist = r_T - 2 * A + r_I
            dist = torch.reshape(torch.transpose(dist, 0, 1), shape=(1, sI[1], sI[2], dist.shape[0]))
            # protecting against numerical problems, dist should be positive
            dist = torch.clamp(dist, min=float(0.0))
            # dist = tf.sqrt(dist)
            raw_distances_list += [dist]

        cs_flow.raw_distances = torch.cat(raw_distances_list)

        relative_dist = cs_flow.calc_relative_distances()
        cs_flow.__calculate_CS(relative_dist)
        return cs_flow

    # --
    @staticmethod
    def create_using_L1(I_features, T_features, sigma=float(0.5), b=float(1.0)):
        cs_flow = CSFlow(sigma, b)
        sT = T_features.shape
        sI = I_features.shape

        Ivecs = torch.reshape(I_features, (sI[0], -1, sI[3]))
        Tvecs = torch.reshape(T_features, (sI[0], -1, sT[3]))
        raw_distances_list = []
        for i in range(sT[0]):
            Ivec, Tvec = Ivecs[i], Tvecs[i]
            dist = torch.abs(torch.sum(Ivec.unsqueeze(1) - Tvec.unsqueeze(0), dim=2))
            dist = torch.reshape(torch.transpose(dist, 0, 1), shape=(1, sI[1], sI[2], dist.shape[0]))
            # protecting against numerical problems, dist should be positive
            dist = torch.clamp(dist, min=float(0.0))
            # dist = tf.sqrt(dist)
            raw_distances_list += [dist]

        cs_flow.raw_distances = torch.cat(raw_distances_list)

        relative_dist = cs_flow.calc_relative_distances()
        cs_flow.__calculate_CS(relative_dist)
        return cs_flow

    # --
    @staticmethod
    def create_using_dotP(I_features, T_features, sigma=float(0.5), b=float(1.0)):
        cs_flow = CSFlow(sigma, b)
        # prepare feature before calculating cosine distance
        T_features, I_features = cs_flow.center_by_T(T_features, I_features)
        T_features = CSFlow.l2_normalize_channelwise(T_features)
        I_features = CSFlow.l2_normalize_channelwise(I_features)

        # work seperatly for each example in dim 1
        cosine_dist_l = []
        N = T_features.size()[0]
        temp_T_features = T_features.detach()
        temp_I_features = I_features.detach()

        for i in range(N):
            T_features_i = temp_T_features[i, :, :, :].unsqueeze_(0)  # 1HWC --> 1CHW
            I_features_i = temp_I_features[i, :, :, :].unsqueeze_(0).permute((0, 3, 1, 2))
            patches_PC11_i = cs_flow.patch_decomposition(T_features_i)  # 1HWC --> PC11, with P=H*W
            cosine_dist_i = torch.nn.functional.conv2d(I_features_i, patches_PC11_i)
            cosine_dist_1HWC = cosine_dist_i.permute((0, 2, 3, 1))
            cosine_dist_l.append(cosine_dist_i.permute((0, 2, 3, 1)))  # back to 1HWC

        cs_flow.cosine_dist = torch.cat(cosine_dist_l, dim=0)
        cs_flow.raw_distances = - (cs_flow.cosine_dist - 1) / 2  ### why -

        relative_dist = cs_flow.calc_relative_distances()
        cs_flow.__calculate_CS(relative_dist)
        return cs_flow

    def calc_relative_distances(self, axis=TensorAxis.C):
        epsilon = 1e-5
        div = torch.min(self.raw_distances, dim=axis, keepdim=True)[0]
        relative_dist = self.raw_distances / (div + epsilon)
        return relative_dist

    @staticmethod
    def sum_normalize(cs, axis=TensorAxis.C):
        reduce_sum = torch.sum(cs, dim=axis, keepdim=True)
        cs_normalize = torch.div(cs, reduce_sum)
        return cs_normalize

    def center_by_T(self, T_features, I_features):
        # assuming both input are of the same size
        # calculate stas over [batch, height, width], expecting 1x1xDepth tensor
        axes = [0, 1, 2]
        self.meanT = T_features.mean(0, keepdim=True).mean(1, keepdim=True).mean(2, keepdim=True)
        self.varT = T_features.var(0, keepdim=True).var(1, keepdim=True).var(2, keepdim=True)
        self.T_features_centered = T_features - self.meanT
        self.I_features_centered = I_features - self.meanT

        return self.T_features_centered, self.I_features_centered

    @staticmethod
    def l2_normalize_channelwise(features):
        norms = features.norm(p=2, dim=TensorAxis.C, keepdim=True)
        features = features.div(norms)
        return features

    def patch_decomposition(self, T_features):
        # 1HWC --> 11PC --> PC11, with P=H*W
        (N, H, W, C) = T_features.shape
        P = H * W
        patches_PC11 = T_features.reshape(shape=(1, 1, P, C)).permute(dims=(2, 3, 0, 1))
        return patches_PC11

    @staticmethod
    def pdist2(x, keepdim=False):
        sx = x.shape
        x = x.reshape(shape=(sx[0], sx[1] * sx[2], sx[3]))
        differences = x.unsqueeze(2) - x.unsqueeze(1)
        distances = torch.sum(differences**2, -1)
        if keepdim:
            distances = distances.reshape(shape=(sx[0], sx[1], sx[2], sx[3]))
        return distances

    @staticmethod
    def calcR_static(sT, order='C', deformation_sigma=0.05):
        # oreder can be C or F (matlab order)
        pixel_count = sT[0] * sT[1]

        rangeRows = range(0, sT[1])
        rangeCols = range(0, sT[0])
        Js, Is = np.meshgrid(rangeRows, rangeCols)
        row_diff_from_first_row = Is
        col_diff_from_first_col = Js

        row_diff_from_first_row_3d_repeat = np.repeat(row_diff_from_first_row[:, :, np.newaxis], pixel_count, axis=2)
        col_diff_from_first_col_3d_repeat = np.repeat(col_diff_from_first_col[:, :, np.newaxis], pixel_count, axis=2)

        rowDiffs = -row_diff_from_first_row_3d_repeat + row_diff_from_first_row.flatten(order).reshape(1, 1, -1)
        colDiffs = -col_diff_from_first_col_3d_repeat + col_diff_from_first_col.flatten(order).reshape(1, 1, -1)
        R = rowDiffs ** 2 + colDiffs ** 2
        R = R.astype(np.float32)
        R = np.exp(-(R) / (2 * deformation_sigma ** 2))
        return R

# --------------------------------------------------
#           CX loss
# --------------------------------------------------


def CX_loss(T_features, I_features, deformation=False, dis=False):
    # T_features = tf.convert_to_tensor(T_features, dtype=tf.float32)
    # I_features = tf.convert_to_tensor(I_features, dtype=tf.float32)
    # since this is a convertion of tensorflow to pytorch we permute the tensor from
    # T_features = normalize_tensor(T_features)
    # I_features = normalize_tensor(I_features)

    # since this originally Tensorflow implemntation
    # we modify all tensors to be as TF convention and not as the convention of pytorch.
    # def from_pt2tf(Tpt):
    #     Ttf = Tpt.permute(0, 2, 3, 1)
    #     return Ttf
    # N x C x H x W --> N x H x W x C
    # T_features_tf = from_pt2tf(T_features)
    # I_features_tf = from_pt2tf(I_features)

    # cs_flow = CSFlow.create_using_dotP(I_features_tf, T_features_tf, sigma=1.0)
    T_features = T_features.permute(0, 2, 3, 1)
    I_features = I_features.permute(0, 2, 3, 1)

    # cs_flow = CSFlow.create_using_L2(I_features, T_features, sigma=1.0)
    cs_flow = CSFlow.create_using_L1(I_features, T_features, sigma=1.0)
    # cs_flow = CSFlow.create_using_dotP(I_features, T_features)

    # sum_normalize:
    # To:
    cs = cs_flow.cs_NHWC

    if deformation:
        deforma_sigma = 0.001
        sT = T_features.shape[1:2 + 1]
        R = CSFlow.calcR_static(sT, deformation_sigma=deforma_sigma)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cs *= torch.Tensor(R).unsqueeze(dim=0).to(device)

    if dis:
        CS = []
        k_max_NC = torch.max(torch.max(cs, dim=1)[1], dim=1)[1]
        indices = k_max_NC.cpu()
        N, C = indices.shape
        for i in range(N):
            CS.append((C - len(torch.unique(indices[i, :]))) / C)
        score = torch.FloatTensor(CS)
    else:
        # reduce_max X and Y dims
        # cs = CSFlow.pdist2(cs,keepdim=True)
        k_max_NC = torch.max(torch.max(cs, dim=1)[0], dim=1)[0]
        # reduce mean over C dim
        CS = torch.mean(k_max_NC, dim=1)
        # score = 1/CS
        # score = torch.exp(-CS*10)
        score = -torch.log(CS)

    # reduce mean over N dim
    score = torch.mean(score)
    return score


def symetric_CX_loss(T_features, I_features):
    score = (CX_loss(T_features, I_features) + CX_loss(I_features, T_features)) / 2
    return score


def evaluation_criteria(pre, label):
    pre = np.array(pre)
    label = np.array(label)

    srcc = stats.spearmanr(pre, label)[0]
    plcc = stats.pearsonr(pre, label)[0]
    krcc = stats.stats.kendalltau(pre, label)[0]
    rmse = np.sqrt(((pre - label) ** 2).mean())
    mae = np.abs((pre - label)).mean()

    return srcc, plcc, krcc, rmse, mae


def mos_rescale(mos, min_val, max_val, scale_min=0, scale_max=1):
    mos = scale_min + (mos - min_val) * ((scale_max - scale_min) / (max_val - min_val))

    return mos


def ranking_loss(pre, label, loss):
    rank_loss = 0.0
    rank_id = [(i, j) for i in range(len(pre)) for j in range(len(pre)) if i != j and i <= j]
    for i in range(len(rank_id)):
        pre_1 = pre[rank_id[i][0]]
        pre_2 = pre[rank_id[i][1]]
        label_1 = label[rank_id[i][0]]
        label_2 = label[rank_id[i][1]]
        rank_loss += loss(pre_1 - pre_2, label_1 - label_2)

    if len(pre) != 1:
        rank_loss /= (len(pre) * (len(pre) - 1) / 2)

    return rank_loss


def relative_ranking_loss(pre, label):
    # Relative Ranking Loss
    sort_index = [x for _, x in sorted(zip(pre, list(range(len(pre)))), reverse=True)]
    high_pre = pre[sort_index[0]]
    second_high_pre = pre[sort_index[1]]
    low_pre = pre[sort_index[-1]]
    second_low_pre = pre[sort_index[-2]]

    high_label = label[sort_index[0]]
    second_high_label = label[sort_index[1]]
    low_label = label[sort_index[-1]]
    second_low_label = label[sort_index[-2]]

    margin1 = second_high_label - low_label
    margin2 = high_label - second_low_label

    triplet_loss_1 = abs(high_pre - second_high_pre) - abs(high_pre - low_pre) + margin1
    triplet_loss_2 = abs(second_low_pre - low_pre) - abs(high_pre - low_pre) + margin2

    if triplet_loss_1 <= 0:
        triplet_loss_1 = 0

    if triplet_loss_2 <= 0:
        triplet_loss_2 = 0

    rank_loss = triplet_loss_1 + triplet_loss_2

    return rank_loss


def pseudo_huber_loss(pre, label, delta):
    # loss = (delta ** 2) * (torch.sqrt(1 + torch.square((pre - label) / (delta + 1e-8))) - 1)
    loss = (delta ** 2) * ((1 + ((pre - label) / (delta + 1e-8)) ** 2) ** (1 / 2) - 1)

    return loss


k = np.float32([1, 4, 6, 4, 1])
k = np.outer(k, k)
kern = k / k.sum()


def local_normalize(img, num_ch=1, const=127.0):
    if num_ch == 1:
        mu = convolve(img[:, :, 0], kern, mode='nearest')
        mu_sq = mu * mu
        im_sq = img[:, :, 0] * img[:, :, 0]
        tmp = convolve(im_sq, kern, mode='nearest') - mu_sq
        sigma = np.sqrt(np.abs(tmp))
        structdis = (img[:, :, 0] - mu) / (sigma + const)

        # Rescale within 0 and 1
        # structdis = (structdis + 3) / 6
        structdis = 2. * structdis / 3.
        norm = structdis[:, :, None]

    elif num_ch > 1:
        norm = np.zeros(img.shape, dtype='float32')
        for ch in range(num_ch):
            mu = convolve(img[:, :, ch], kern, mode='nearest')
            mu_sq = mu * mu
            im_sq = img[:, :, ch] * img[:, :, ch]
            tmp = convolve(im_sq, kern, mode='nearest') - mu_sq
            sigma = np.sqrt(np.abs(tmp))
            structdis = (img[:, :, ch] - mu) / (sigma + const)

            # Rescale within 0 and 1
            # structdis = (structdis + 3) / 6
            structdis = 2. * structdis / 3.
            norm[:, :, ch] = structdis

    return norm


class LowerBound(Function):
    def forward(ctx, inputs, bound):
        b = torch.ones(inputs.size()) * bound
        b = b.to(inputs.device)
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors

        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


class GDN(nn.Module):
    """Generalized divisive normalization layer.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))
    """
    def __init__(self,
                 ch,
                 device,
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=.1,
                 reparam_offset=2 ** -18):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = torch.FloatTensor([reparam_offset])

        self.build(ch, torch.device(device))

    def build(self, ch, device):
        self.pedestal = self.reparam_offset ** 2
        self.beta_bound = (self.beta_min + self.reparam_offset ** 2) ** .5
        self.gamma_bound = self.reparam_offset

        # Create beta param
        beta = torch.sqrt(torch.ones(ch) + self.pedestal)
        self.beta = nn.Parameter(beta.to(device))

        # Create gamma param
        eye = torch.eye(ch)
        g = self.gamma_init * eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)

        self.gamma = nn.Parameter(gamma.to(device))
        self.pedestal = self.pedestal.to(device)

    def forward(self, inputs):
        device_id = inputs.device.index

        beta = self.beta.to(device_id)
        gamma = self.gamma.to(device_id)
        pedestal = self.pedestal.to(device_id)

        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size()
            inputs = inputs.view(bs, ch, d * w, h)

        _, ch, _, _ = inputs.size()

        # Beta bound and reparam
        beta = LowerBound()(beta, self.beta_bound)
        beta = beta ** 2 - pedestal

        # Gamma bound and reparam
        gamma = LowerBound()(gamma, self.gamma_bound)
        gamma = gamma ** 2 - pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(inputs ** 2, gamma, beta)
        norm_ = torch.sqrt(norm_)

        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs


class L2pooling(nn.Module):
    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
        super(L2pooling, self).__init__()

        self.padding = (filter_size - 2) // 2
        self.stride = stride
        self.channels = channels

        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:, None] * a[None, :])
        g = g / torch.sum(g)
        self.register_buffer('filter', g[None, None, :, :].repeat((self.channels, 1, 1, 1)))

    def forward(self, input):
        input = input ** 2
        out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])

        return (out + 1e-12).sqrt()


class group_norm(torch.nn.Module):
    def __init__(self, dim_to_norm=None, dim_hidden=16, num_nodes=None, num_groups=None, skip_weight=None, **w):
        super(group_norm, self).__init__()
        self.num_nodes = num_nodes
        self.num_groups = num_groups
        self.skip_weight = skip_weight
        self.dim_hidden = dim_hidden

        self.bn = torch.nn.BatchNorm1d(dim_hidden * self.num_groups * self.num_nodes, momentum=0.3)
        self.group_func = torch.nn.Linear(dim_hidden, self.num_groups, bias=True)

    def forward(self, x):
        if self.num_groups == 1:
            x_temp = self.bn(x)
        else:
            score_cluster = F.softmax(self.group_func(x), dim=2)
            x_temp = torch.cat([score_cluster[:, :, group].unsqueeze(dim=2) * x for group in range(self.num_groups)], dim=2)
            # batch, number_nodes, num_groups * dim_hidden
            x_temp = self.bn(x_temp.view(-1, self.num_nodes * self.num_groups * self.dim_hidden))
            x_temp = x_temp.view(-1, self.num_nodes, self.num_groups, self.dim_hidden).sum(dim=2)

        x = x + x_temp * self.skip_weight

        return x


def node_norm(x, p=2):
    """
    :param x: [batch, n_nodes, features]
    :return:
    """
    std_x = torch.std(x, dim=2, keepdim=True)
    x = x / (std_x ** (1 / p) + 1e-5)

    return x


def one_zero_normalization(x):
    if x.dim() == 4:
        dim_0 = x.size(0)
        dim_1 = x.size(1)
        dim_2 = x.size(2)
        dim_3 = x.size(3)

        x = x.view(dim_0, -1)
        x = x - x.min(dim=1, keepdim=True)[0]
        x = x / x.max(dim=1, keepdim=True)[0]
        x = x.view(-1, dim_1, dim_2, dim_3)

    elif x.dim() == 3:
        dim_0 = x.size(0)
        dim_1 = x.size(1)
        dim_2 = x.size(2)

        x = x.view(dim_0, -1)
        x = x - x.min(dim=1, keepdim=True)[0]
        x = x / x.max(dim=1, keepdim=True)[0]
        x = x.view(-1, dim_1, dim_2)

    return x


def mean_std_normalization(x):
    if x.dim() == 4:
        dim_0 = x.size(0)
        dim_1 = x.size(1)
        dim_2 = x.size(2)
        dim_3 = x.size(3)

        x = x.view(dim_0, -1)
        x = x - x.mean(dim=1, keepdim=True)
        x = x / (x.std(dim=1, keepdim=True) + 1e-12)
        x = x.view(-1, dim_1, dim_2, dim_3)

    elif x.dim() == 3:
        dim_0 = x.size(0)
        dim_1 = x.size(1)
        dim_2 = x.size(2)

        x = x.view(dim_0, -1)
        x = x - x.mean(dim=1, keepdim=True)
        x = x / (x.std(dim=1, keepdim=True) + 1e-12)
        x = x.view(-1, dim_1, dim_2)

    else:
        x = x - x.mean(dim=1, keepdim=True)
        x = x / (x.std(dim=1, keepdim=True) + 1e-12)

    return x


def vec_l2_norm(x):
    """
    :param x: [Batch_size, num_feature]
    :return vector after L2 Normalization: [Batch_size, num_feature]
    """
    # x: [Batch_size, num_feature]
    if x.dim() == 2:
        norm = x.norm(p=2, dim=1, keepdim=True) + 1e-5

    # x: [Batch_size, num_node, num_feature]
    elif x.dim() == 3:
        norm = x.norm(p=2, dim=2, keepdim=True) + 1e-5

    l2_norm = x.div(norm)

    return l2_norm


def bilinear_pool(feature_1, feature_2):
    """
    :param feature_1: [B, C1, H, W]
    :param feature_2: [B, C2, H, W]
    :return bilinear pooling vector: [Batch_size, C1 * C2]
    """
    num_feature_1 = feature_1.size()[1]
    num_feature_2 = feature_2.size()[1]
    H = feature_1.size()[2]
    W = feature_1.size()[3]

    feature_1 = feature_1.view(-1, num_feature_1, H * W)
    feature_2 = feature_2.view(-1, num_feature_2, H * W)
    X = torch.bmm(feature_1, torch.transpose(feature_2, 1, 2)) / (H * W)  # Bilinear
    X = X.view(-1, num_feature_1 * num_feature_2)
    X = torch.nn.functional.normalize(torch.sign(X) * torch.sqrt(torch.abs(X) + 1e-8))

    return X


def gaussian_prior(mean, scale):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noise = torch.randn(mean.size()).to(device)
    mos_pred = mean + noise * scale

    return mos_pred


def mkdirs(path):
    os.makedirs(path, exist_ok=True)


# def setup_seed(seed):
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


def getFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if os.path.splitext(i)[1] == suffix:
            filename.append(i)
    return filename


def getTIDFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if suffix.find(os.path.splitext(i)[1]) != -1:
            filename.append(i[1:3])
    return filename


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# def image_show(path):
#     image = mpimg.imread(path)
#     plt.imshow(image)
#     plt.axis('off')
#     plt.show()


def image_tensor_show(image_tensor):
    for i in range(np.shape(image_tensor)[0]):
        temp = image_tensor[i]
        temp = np.squeeze(temp, axis=0)
        temp = np.transpose(temp, (1, 2, 0))
        plt.imshow(temp)
        plt.axis('off')
        plt.show()


class SimpleProgressBar:
    def __init__(self, total_len, pat='#', show_step=False, print_freq=1):
        self.len = total_len
        self.pat = pat
        self.show_step = show_step
        self.print_freq = print_freq
        self.out_stream = stdout

    def show(self, cur, desc):
        bar_len, _ = shutil.get_terminal_size()
        # The tab between desc and the progress bar should be counted.
        # And the '|'s on both ends be counted, too
        bar_len = bar_len - self.len_with_tabs(desc + '\t') - 2
        bar_len = int(bar_len * 0.8)
        cur_pos = int(((cur + 1) / self.len) * bar_len)
        cur_bar = '|' + self.pat * cur_pos + ' ' * (bar_len - cur_pos) + '|'

        disp_str = "{0}\t{1}".format(desc, cur_bar)

        # Clean
        self.write('\033[K')

        if self.show_step and (cur % self.print_freq) == 0:
            self.write(disp_str, new_line=True)
            return

        if (cur + 1) < self.len:
            self.write(disp_str)
        else:
            self.write(disp_str, new_line=True)

        self.out_stream.flush()

    @staticmethod
    def len_with_tabs(s):
        return len(s.expandtabs())

    def write(self, content, new_line=False):
        end = '\n' if new_line else '\r'
        self.out_stream.write(content + end)
