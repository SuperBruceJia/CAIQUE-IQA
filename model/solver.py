# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import random

import cv2
import os
import time
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
import torchvision.transforms as T

from model.network import Network
from data_process.load_data import data_loader
from lib.superpixel import superpixel
from lib.utils import evaluation_criteria, PLCC_loss, image_show
from lib.image_process import inference_crop_patches


class Solver(object):
    """The solver for training, validating, and testing the NLNet"""
    def __init__(self, args):
        self.exp_id = args.exp_id
        self.batch_size = args.batch_size
        self.database = args.database
        self.save_model_path = args.save_model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.epochs = args.epochs
        self.lr = args.lr
        self.lr_decay_ratio = args.lr_decay_ratio
        self.lr_decay_epoch = args.lr_decay_epoch
        self.weight_decay = args.weight_decay

        self.num_dis_type = args.num_dis_type

        # Choose which model to train
        self.model = Network(args).to(self.device)
        if args.multi_gpu is True:
            self.model = nn.DataParallel(self.model)
        self.model.train(True)

        # Print Model Architecture
        # print('*' * 100)
        # print(self.model)
        # print('*' * 100, '\n')

        if args.multi_gpu is False:
            paras = self.model.parameters()
            for name, param in self.model.named_parameters():
                if 'cnn_backbone' in name:
                    param.requires_grad = False
        else:
            paras = self.model.module.parameters()
            for name, param in self.model.module.named_parameters():
                if 'cnn_backbone' in name:
                    param.requires_grad = False

        # Print trainable model parameters
        # print('*' * 100)
        # print('The trained parameters are as follows: \n')
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(name, ', with shape: ', np.shape(param))
        # print('*' * 100, '\n')

        # Loss Functions
        # self.quality_loss = nn.SmoothL1Loss().to(self.device)
        # self.quality_div_loss = nn.SmoothL1Loss().to(self.device)
        # self.quality_rank_loss = nn.SmoothL1Loss().to(self.device)
        # self.quality_residual_loss = nn.SmoothL1Loss().to(self.device)
        # self.dis_type_loss = nn.CrossEntropyLoss().to(self.device)

        # Optimizer
        self.solver = optim.AdamW(params=filter(lambda p: p.requires_grad, paras),
                                  lr=self.lr,
                                  weight_decay=self.weight_decay)

    def train(self, args, pre_srcc, pre_test_srcc, pre_test_plcc, pre_test_krcc, pre_test_mse, pre_test_mae):
        # Varied patch size input
        self.patch_size = args.patch_size

        # Get training, and testing data
        self.train_loader, self.val_loader, self.test_loader = data_loader(args)

        pre_model = self.save_model_path + self.database + '-' + str(self.exp_id) + '.pth'
        if os.path.exists(pre_model):
            self.model.load_state_dict(torch.load(pre_model))
            # print('Loaded previous trained model. Current patch size:', self.patch_size)
            self.model.train(True)

            for name, param in self.model.named_parameters():
                if 'cnn_backbone' in name:
                    param.requires_grad = True

        test_srcc = pre_test_srcc
        test_plcc = pre_test_plcc
        test_krcc = pre_test_krcc
        test_rmse = pre_test_mse
        test_mae = pre_test_mae

        print('PATCH_SIZE Epoch TRAINING Loss\t TRAINING SRCC PLCC KRCC MSE MAE\t '
              'VALIDATION SRCC PLCC KRCC MSE MAE\t TESTING SRCC PLCC KRCC MSE MAE\t')
        for t in range(1, self.epochs + 1):
            epoch_loss = []
            pred_scores = []
            gt_scores = []
            for i, (patch, label) in enumerate(self.train_loader):
                # [batch_size, num_patch, 3, patch_size, patch_size]
                patch = torch.as_tensor(patch.to(self.device))
                num_batch = np.shape(patch)[0]
                num_patch = np.shape(patch)[1]
                # [batch_size * num_patch, 3, patch_size, patch_size]
                patch = patch.reshape([-1, 3, self.patch_size, self.patch_size])

                # [batch_size, 1]
                label = torch.as_tensor(label.to(self.device), dtype=torch.float32)
                label = torch.reshape(label, [-1])  # [batch_size]

                # dis_type = torch.as_tensor(dis_type.to(self.device), dtype=torch.long)  # [batch_size, 1]
                # dis_type = torch.reshape(dis_type, [-1])  # [batch_size]
                # dis_type = dis_type.repeat(num_patch)  # [batch_size * num_patch]

                # self.solver.zero_grad(set_to_none=True)
                self.solver.zero_grad()
                # pre_raw, pre_nl, pre_l, type_pre = self.model(patch)
                pre_raw, pre_nl, pre_l = self.model(patch)
                pre_raw = torch.reshape(pre_raw, [num_batch, num_patch]).mean(dim=1)
                pre_nl = torch.reshape(pre_nl, [num_batch, num_patch]).mean(dim=1)
                pre_l = torch.reshape(pre_l, [num_batch, num_patch]).mean(dim=1)
                pre = (pre_raw + pre_l) / 2 + pre_nl

                pred_scores += pre.cpu().tolist()
                gt_scores += label.cpu().tolist()

                # Quality Regression Loss
                quality_loss = PLCC_loss(pre, label)
                # quality_loss = self.quality_loss(pre, label)

                # Divergence Loss
                quality_div_loss = PLCC_loss(pre_raw, pre_l)
                quality_residual_loss = (PLCC_loss(pre_raw - label, pre_nl)
                                         + PLCC_loss(pre - label, pre_nl)
                                         + PLCC_loss(pre_l - label, pre_nl)) / 3
                # quality_div_loss = self.quality_div_loss(pre_raw, pre_l)
                # quality_residual_loss = self.quality_residual_loss(pre_raw - label, pre_nl)

                # Overall Loss
                # loss = quality_loss + quality_div_loss + quality_residual_loss + rank_loss
                # loss = quality_loss + quality_div_loss + quality_residual_loss + dis_type_loss
                loss = quality_loss + quality_div_loss + quality_residual_loss
                epoch_loss.append(loss.item())
                loss.backward()
                self.solver.step()

            train_srcc, train_plcc, train_krcc, train_rmse, train_mae \
                = evaluation_criteria(pre=pred_scores, label=gt_scores)

            if t % 20 == 0:
                # performance on the validation set
                val_srcc, val_plcc, val_krcc, val_rmse, val_mae = self.validate_test(self.val_loader)
                if val_srcc >= pre_srcc:
                    pre_srcc = val_srcc

                    # Get the performance on the testing set w.r.t. the best validation performance
                    test_srcc, test_plcc, test_krcc, test_rmse, test_mae = self.validate_test(self.test_loader)
                    torch.save(self.model.state_dict(),
                               self.save_model_path + self.database + '-' + str(self.exp_id) + '.pth')

                print("%d, %d, %4.3f || "
                      "%4.3f, %4.3f, %4.3f, %4.3f, %4.3f || "
                      "%4.3f, %4.3f, %4.3f, %4.3f, %4.3f || "
                      "%4.3f, %4.3f, %4.3f, %4.3f, %4.3f"
                      % (self.patch_size, t, sum(epoch_loss) / len(epoch_loss),
                         train_srcc, train_plcc, train_krcc, train_rmse, train_mae,
                         val_srcc, val_plcc, val_krcc, val_rmse, val_mae,
                         test_srcc, test_plcc, test_krcc, test_rmse, test_mae))

        return test_srcc, test_plcc, test_krcc, test_rmse, test_mae, pre_srcc

    def validate_test(self, data):
        """Validation and Testing"""
        pred_scores = []
        gt_scores = []

        with torch.no_grad():
            for img, label in data:
                label = torch.as_tensor(label.to(self.device), dtype=torch.float32)  # [batch_size, 1]
                label = torch.reshape(label, [-1])  # [batch_size]

                img = img.squeeze(dim=0)
                transform = T.ToPILImage()
                im = transform(img)
                img_numpy = np.array(im)
                w, h = im.size

                centroids, distances = superpixel(img_numpy)
                predictions = []
                for i in range(len(distances)):
                    cx, cy = centroids[i][0], centroids[i][1]
                    dis = distances[i]
                    half_dis = dis / 2

                    # Left Top Corner
                    if cx < half_dis and cy < half_dis:
                        cropped_patch = im.crop((0, 0, dis, dis))
                    # Right Bottom Corner
                    elif cx + half_dis > h and cy + half_dis > w:
                        cropped_patch = im.crop((w - dis, h - dis, w, h))
                    # Right Top Corner
                    elif cx < half_dis and cy + half_dis > w:
                        cropped_patch = im.crop((w - dis, 0, w, dis))
                    # Left Bottom Corner
                    elif cx + half_dis > h and cy < half_dis:
                        cropped_patch = im.crop((0, h - dis, dis, h))
                    # [Center] Left
                    elif cx > half_dis and cx + half_dis < h and cy < half_dis:
                        cropped_patch = im.crop((0, cx - half_dis, dis, cx + half_dis))
                    # [Center] Top
                    elif cx < half_dis and cy > half_dis and cy + half_dis < w:
                        cropped_patch = im.crop((cy - half_dis, 0, cy + half_dis, dis))
                    # [Center] right
                    elif cx > half_dis and cx + half_dis < h and cy + half_dis > w:
                        cropped_patch = im.crop((w - dis, cx - half_dis, w, cx + half_dis))
                    # [Center] Bottom
                    elif cx + half_dis > h and cy > half_dis and cy + half_dis < w:
                        cropped_patch = im.crop((cy - half_dis, h - dis, cy + half_dis, h))
                    # Others
                    else:
                        cropped_patch = im.crop((cy - half_dis, cx - half_dis, cy + half_dis, cx + half_dis))

                    patch = inference_crop_patches(cropped_patch)
                    patch = torch.as_tensor(patch.to(self.device))  # [1, 3, patch_size, patch_size]

                    if patch.dim() != 4:
                        patch = patch.unsqueeze(dim=0)  # [1, 3, patch_size, patch_size]

                    # pre_raw, pre_nl, pre_l, _ = self.model(patch)
                    pre_raw, pre_nl, pre_l = self.model(patch)
                    pre_raw = torch.reshape(pre_raw, [-1]).cpu().detach().numpy()
                    pre_nl = torch.reshape(pre_nl, [-1]).cpu().detach().numpy()
                    pre_l = torch.reshape(pre_l, [-1]).cpu().detach().numpy()
                    # pre = (pre_raw + pre_nl + pre_l) / 3
                    pre = (pre_raw + pre_l) / 2 + pre_nl
                    predictions += pre.tolist()

                pred_scores.append(sum(predictions) / len(predictions))
                gt_scores += label.cpu().tolist()

        srcc, plcc, krcc, rmse, mae = evaluation_criteria(pre=pred_scores, label=gt_scores)

        return srcc, plcc, krcc, rmse, mae
