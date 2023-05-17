# -*- coding: utf-8 -*-

import cv2
import time
import random
import numpy as np
from numba import jit

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision.transforms.functional import to_tensor

from lib.utils import patch_similarity
from lib.superpixel import superpixel


def inference_crop_patches(patch):
    cnn_patch = to_tensor(patch)
    cnn_patch = val_test_transforms(cnn_patch)

    return cnn_patch


def slide_crop_patches(im, args=None):
    """Non-overlapping Cropped Patches"""
    if args != None:
        patch_size = args.patch_size
    else:
        patch_size = 224

    slide_size = 64
    w, h = im.size

    cnn_patches = ()
    for i in range(0, h - patch_size, slide_size):
        for j in range(0, w - patch_size, slide_size):
            cropped_patch = im.crop((j, i, j + patch_size, i + patch_size))
            cnn_patch = to_tensor(cropped_patch)
            cnn_patch = val_test_transforms(cnn_patch)
            cnn_patches += (cnn_patch,)

    if (w - patch_size) % slide_size != 0:
        for i in range(0, h - patch_size, slide_size):
            cropped_patch = im.crop((w - patch_size, i, w, i + patch_size))
            cnn_patch = to_tensor(cropped_patch)
            cnn_patch = val_test_transforms(cnn_patch)
            cnn_patches += (cnn_patch,)

    if (h - patch_size) % slide_size != 0:
        for j in range(0, w - patch_size, slide_size):
            cropped_patch = im.crop((j, h - patch_size, j + patch_size, h))
            cnn_patch = to_tensor(cropped_patch)
            cnn_patch = val_test_transforms(cnn_patch)
            cnn_patches += (cnn_patch,)

    if (w - patch_size) % slide_size != 0 and (h - patch_size) % slide_size != 0:
        cropped_patch = im.crop((w - patch_size, h - patch_size, w, h))
        cnn_patch = to_tensor(cropped_patch)
        cnn_patch = val_test_transforms(cnn_patch)
        cnn_patches += (cnn_patch,)

    return torch.stack(cnn_patches)


def random_crop_patches(im, args=None, n_patches=25, train=True):
    w, h = im.size
    patch_size = args.patch_size

    # Get random patch for external learning
    random_patch = ()
    for i in range(n_patches):
        w1 = np.random.randint(low=0, high=w - patch_size + 1)
        h1 = np.random.randint(low=0, high=h - patch_size + 1)

        # Get external random patch
        cropped_patch = im.crop((w1, h1, w1 + patch_size, h1 + patch_size))
        cropped_patch = to_tensor(cropped_patch)
        if train:
            cropped_patch = train_transforms(cropped_patch)
        else:
            cropped_patch = val_test_transforms(cropped_patch)

        random_patch += (cropped_patch,)

    random_patch = torch.stack(random_patch)

    return random_patch


train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

val_test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
