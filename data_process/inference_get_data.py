# -*- coding: utf-8 -*-

import cv2
import time
import os
import h5py
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import models, transforms
from torchvision.transforms.functional import to_tensor

from lib.make_index import make_index, default_loader
from lib.image_process import random_crop_patches, slide_crop_patches
from lib.utils import mos_rescale, local_normalize
from lib.superpixel import superpixel


class IQADataset(Dataset):
    """
    IQA Dataset
    """

    def __init__(self, args, status='train', loader=default_loader):
        """
        :param args: arguments of the model
        :param status: train/val/test
        :param loader: image loader
        """
        self.status = status
        self.loader = loader

        self.args = args
        self.database = args.database

        self.n_patches_train = args.n_patches_train
        self.n_patches_test = args.n_patches_test

        Info = h5py.File(args.data_info, 'r')
        index = Info['index']
        index = index[:, 0 % index.shape[1]]
        ref_ids = Info['ref_ids'][0, :]

        # Get dataset index
        trainindex, valindex, testindex = make_index(args=args, index=index)

        # Split Database and make sure there are no contents overlap
        train_index, val_index, test_index = [], [], []
        for i in range(len(ref_ids)):
            if ref_ids[i] in testindex:
                test_index.append(i)

            elif ref_ids[i] in valindex:
                val_index.append(i)

            else:
                train_index.append(i)

        if 'train' in status:
            self.index = train_index

        if 'val' in status:
            self.index = val_index

        if 'test' in status:
            self.index = test_index

        self.mos = Info['subjective_scores'][0, self.index]
        im_names = [Info[Info['im_names'][0, :][i]][()].tobytes()[::2].decode() for i in self.index]

        # Get image names and their scores
        self.im_names = []
        self.label = []
        for idx in range(len(self.index)):
            self.im_names.append(os.path.join(args.im_dir, im_names[idx]))

            # Scaled MOS
            scale_min = 0
            scale_max = 100
            if self.database == 'CSIQ':
                scaled_mos = mos_rescale(mos=self.mos[idx], min_val=0, max_val=1, scale_min=scale_min, scale_max=scale_max)
            elif self.database == 'TID2013':
                scaled_mos = mos_rescale(mos=self.mos[idx], min_val=0, max_val=9, scale_min=scale_min, scale_max=scale_max)
            elif self.database == 'KonIQ' or self.database == 'KADID':
                scaled_mos = mos_rescale(mos=self.mos[idx], min_val=1, max_val=5, scale_min=scale_min, scale_max=scale_max)
            elif self.database == 'BID':
                scaled_mos = mos_rescale(mos=self.mos[idx], min_val=0, max_val=5, scale_min=scale_min, scale_max=scale_max)
            else:
                scaled_mos = self.mos[idx]
            self.label.append(scaled_mos)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        im = to_tensor(self.loader(self.im_names[idx]))

        # Get labels
        label = torch.as_tensor([self.label[idx], ])

        return im, label
