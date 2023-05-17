# -*- coding: utf-8 -*-

import math
import numbers
from pdb import set_trace as stx
import numpy as np
from einops import rearrange

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torch.optim as optim


class SPPLayer(torch.nn.Module):
    def __init__(self, num_levels):
        super(SPPLayer, self).__init__()

        self.num_levels = num_levels

    def forward(self, x):
        num, c, h, w = x.size()
        for i in range(self.num_levels):
            level = i + 1

            kernel_size = (math.ceil(h / level), math.ceil(w / level))
            stride = (math.ceil(h / level), math.ceil(w / level))
            pooling = (math.floor((kernel_size[0] * level - h + 1) / 2),
                       math.floor((kernel_size[1] * level - w + 1) / 2))
            # tensor = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)
            max_ = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling)
            min_ = -F.max_pool2d(-x, kernel_size=kernel_size, stride=stride, padding=pooling)
            mean_ = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling)
            std_ = torch.sqrt(F.relu(F.avg_pool2d(torch.pow(x, 2),
                                                  kernel_size=kernel_size,
                                                  stride=stride,
                                                  padding=pooling) - torch.pow(mean_, 2)))
            tensor = torch.cat((max_, min_, mean_, std_), 1).view(num, -1)

            if i == 0:
                x_flatten = tensor.view(num, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)

        return x_flatten


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

        return (out + 1e-5).sqrt()


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


# Layer Norm
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1
        
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)

        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        
        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)

        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNormType):
        super(LayerNorm, self).__init__()

        if LayerNormType == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]

        return to_4d(self.body(to_3d(x)), h, w)


# Feed-Forward Network (FN)
class FeedForward(nn.Module):
    def __init__(self, dim, ratio, bias):
        super(FeedForward, self).__init__()

        self.hidden = int(dim * ratio)

        self.project_in_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_in_2 = nn.Conv2d(dim, dim, kernel_size=3,
                                      padding=1, padding_mode='zeros', bias=bias)
        self.project_in_3 = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)

        self.project_out_1 = nn.Conv2d(dim, self.hidden, kernel_size=1, bias=bias)
        self.project_out_2 = nn.Conv2d(self.hidden, dim, kernel_size=3,
                                       padding=1, padding_mode='zeros', bias=bias)
        self.project_out_3 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = F.elu(self.project_in_1(x))
        x = F.elu(self.project_in_2(x))
        x = self.project_in_3(x)

        x1, x2 = x.chunk(2, dim=1)
        x = F.elu(x1) * x2

        x = F.elu(self.project_out_1(x))
        x = F.elu(self.project_out_2(x))
        x = self.project_out_3(x)
        
        return x


# Spatial-channel Self-Attention
class Attention(nn.Module):
    def __init__(self, dim, num_heads, ratio, bias):
        super(Attention, self).__init__()

        self.num_heads = num_heads
        self.hidden = int(dim * ratio)

        self.temperature_spatial = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature_channel = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.q_trans = nn.Conv2d(dim, dim, kernel_size=3, stride=1,
                                 padding=1, padding_mode='zeros', bias=bias)
        self.k_trans = nn.Conv2d(dim, dim, kernel_size=3, stride=1,
                                 padding=1, padding_mode='zeros', bias=bias)
        self.v_trans = nn.Conv2d(dim, dim, kernel_size=3, stride=1,
                                 padding=1, padding_mode='zeros', bias=bias)

        self.q_trans_2 = nn.Conv2d(dim, self.num_heads, kernel_size=1, bias=bias)
        self.k_trans_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_trans_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.project_out_1 = nn.Conv2d(self.num_heads, self.hidden, kernel_size=1, bias=bias)
        self.project_out_2 = nn.Conv2d(self.hidden, dim, kernel_size=3,
                                       padding=1, padding_mode='zeros', bias=bias)
        self.project_out_3 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        q = F.elu(self.q(x))
        q = F.elu(self.q_trans(q))
        q = self.q_trans_2(q)
        
        k = F.elu(self.k(x))
        k = F.elu(self.k_trans(k))
        k = F.elu(self.k_trans_2(k))
        
        v = F.elu(self.v(x))
        v = F.elu(self.v_trans(v))
        v = F.elu(self.v_trans_2(v))

        q = rearrange(q, 'b (head m) h w -> b head m (h w)', head=self.num_heads, m=1)  # [b, head, 1, h*w]
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # [b, head, c, h*w]
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # [b, head, c, h*w]
        
        # Spatial-wise Attention -> context modelling
        # Aggregate the features of all positions together to form a global context feature
        spatial_attn = q.softmax(dim=-1)  # [b, head, 1, h*w]

        # Channel-wise Attention -> context mixing
        # capture the channel-wise interdependencies
        # [b, head, 1, h*w] * [b, head, h*w, c] = [b, head, 1, c]
        channel_attn = (spatial_attn @ k.transpose(-2, -1)) * self.temperature_spatial
        channel_attn = channel_attn.softmax(dim=-1)  # [b, head, 1, c]
        # Channel Attention implementing to VALUE
        # [b, head, h*w, c] * [b, head, c, 1] = [b, head, h*w, 1]
        channel_attn = (v.transpose(-2, -1) @ channel_attn.transpose(-2, -1)) * self.temperature_channel
        channel_attn = channel_attn.softmax(dim=-2)

        # Spatial Attention implementing to VALUE
        # # [b, head, 1, h, w]
        # channel_attn = rearrange(channel_attn, 'b head (h w) k -> b head k h w', head=self.num_heads, k=1, h=h, w=w)
        # # [b, head, 1, h, w]
        # spatial_attn = rearrange(spatial_attn, 'b head k (h w) -> b head k h w', head=self.num_heads, k=1, h=h, w=w)
        # [b, head, 1, h, w]
        channel_attn = rearrange(channel_attn, 'b head (h w) k -> b (head k) h w', head=self.num_heads, k=1, h=h, w=w)
        # [b, head, 1, h, w]
        spatial_attn = rearrange(spatial_attn, 'b head k (h w) -> b (head k) h w', head=self.num_heads, k=1, h=h, w=w)
        # [b, head, 1, h, w]
        # channel_attn = rearrange(channel_attn, 'b head (h w) k -> b head k h w', head=self.num_heads, k=1, h=h, w=w)
        # # [b, head, 1, h, w]
        # spatial_attn = rearrange(spatial_attn, 'b head k (h w) -> b head k h w', head=self.num_heads, k=1, h=h, w=w)

        # channel_attn = torch.mean(channel_attn, dim=1, keepdim=False)  # [b, 1, h, w]
        # spatial_attn = torch.mean(spatial_attn, dim=1, keepdim=False)  # [b, 1, h, w]
        # x = spatial_attn + channel_attn + x

        # x: [b, c, h, w] -> [b, head, c, h, w] + [b, head, 1, h, w]
        # attn: [b, head, h, w]
        # out = torch.cat([spatial_attn, channel_attn], dim=1)  # [b, 2 * head, h, w]
        # out = torch.cat([x + spatial_attn.repeat(1, int(c / self.num_heads), 1, 1),
        #                  x + channel_attn.repeat(1, int(c / self.num_heads), 1, 1)], dim=1)
        # out = torch.cat([x.unqueeze(1).repeat(1, self.num_heads, 1, 1, 1) + spatial_attn,
        #                  x.unqueeze(1).repeat(1, self.num_heads, 1, 1, 1) + channel_attn], dim=1)
        # out = rearrange(out, 'b head c h w -> b (head c) h w', head=2 * self.num_heads, c=c, h=h, w=w)
        out = spatial_attn + channel_attn  # [b, head, h, w]
        # x = torch.cat([spatial_attn, channel_attn], dim=1)

        # Transform module -> merge the global context feature into features of all positions
        out = F.elu(self.project_out_1(out))
        out = F.elu(self.project_out_2(out))
        out = self.project_out_3(out)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, bias, LayerNormType, ratio):
        super(TransformerBlock, self).__init__()
        
        self.norm1 = LayerNorm(dim, LayerNormType)
        self.attn = Attention(dim, num_heads, ratio, bias)

        self.norm2 = LayerNorm(dim, LayerNormType)
        self.ffn = FeedForward(dim, ratio, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


# Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=64, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1,
                              padding=1, padding_mode='zeros', bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


# Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        # self.body_0 = nn.Conv2d(n_feat, n_feat * 2, kernel_size=1, stride=1,
        #                         padding=0, padding_mode='replicate', bias=False)
        # self.pooling = L2pooling(channels=n_feat * 2)
        self.pooling = L2pooling(channels=n_feat)
        # self.body_1 = nn.Conv2d(n_feat * 2, n_feat * 2, kernel_size=1, stride=1,
        #                         padding=0, padding_mode='replicate', bias=False)
        
    def forward(self, x):
        # x = self.body_0(x)
        # x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = self.pooling(x)
        # x = self.body_1(x)

        return x


def init_linear(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
