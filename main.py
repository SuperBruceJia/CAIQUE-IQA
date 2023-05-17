#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
import sys
import random
import warnings
import numpy as np
from multiprocessing import cpu_count
from argparse import ArgumentParser

import torch.backends.cudnn

from lib.utils import *
from benchmark.database import database, distortion_type
from model.solver import Solver


def run(args):
    """Run the program"""
    if args.multi_gpu is False:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cpu_num = cpu_count()
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.utils.backcompat.broadcast_warning.enabled = True
    print('Current Train/Validation/Testing database is', args.database, '\n')
    
    # Here we assume to use 10-fold Cross-validation
    k_fold_cv = 10
    srcc_all = np.zeros(k_fold_cv, dtype=float)
    plcc_all = np.zeros(k_fold_cv, dtype=float)
    krcc_all = np.zeros(k_fold_cv, dtype=float)
    rmse_all = np.zeros(k_fold_cv, dtype=float)
    mae_all = np.zeros(k_fold_cv, dtype=float)

    index = 0
    pre_test_srcc = 0.0
    pre_test_plcc = 0.0
    pre_test_krcc = 0.0
    pre_test_mse = 0.0
    pre_test_mae = 0.0

    # Set exp_id from 1 to k_fold_cv
    for id in range(1, k_fold_cv + 1):
        args.exp_id = id
        print('This is the %d training, validation and testing of 10 seeds (from 1 to 10).' % args.exp_id)

        pre_val_srcc = 0.0
        solver = Solver(args=args)
        dis_index = [128, 160, 192, 224, 256, 288, 320]

        # Run the model for 20 times
        for size_id in range(1, 11):
            # Use Random Seed to set up Patch Size
            random.seed(size_id)
            random.shuffle(dis_index)

            random.seed(size_id)
            random_size = random.randint(0, len(dis_index) - 1)
            args.patch_size = dis_index[random_size]

            # Train the model
            srcc_all[index], plcc_all[index], krcc_all[index], rmse_all[index], mae_all[index], best_srcc \
                = solver.train(args=args,
                               pre_srcc=pre_val_srcc,
                               pre_test_srcc=pre_test_srcc,
                               pre_test_plcc=pre_test_plcc,
                               pre_test_krcc=pre_test_krcc,
                               pre_test_mse=pre_test_mse,
                               pre_test_mae=pre_test_mae)

            pre_val_srcc = best_srcc
            pre_test_srcc = srcc_all[index]
            pre_test_plcc = plcc_all[index]
            pre_test_krcc = krcc_all[index]
            pre_test_mse = rmse_all[index]
            pre_test_mae = mae_all[index]

        print('The best testing SRCC: %4.3f, PLCC: %4.3f, KRCC: %4.3f, MSE: %4.3f, MAE: %4.3f \n\n\n\n'
              % (srcc_all[index], plcc_all[index], krcc_all[index], rmse_all[index], mae_all[index]))
        index += 1

    srcc_med = np.median(srcc_all)
    plcc_med = np.median(plcc_all)
    krcc_med = np.median(krcc_all)
    rmse_med = np.median(rmse_all)
    mae_med = np.median(mae_all)
    print('[FINAL RESULT] Testing median SRCC %4.3f, median PLCC %4.3f, '
          'median KRCC %4.3f, median MSE %4.3f,'
          'median MAE %4.3f \n\n\n\n' % (srcc_med, plcc_med, krcc_med, rmse_med, mae_med))


if __name__ == "__main__":
    parser = ArgumentParser(description='`Context-Aware Non-Local Modeling` for `Image Quality Assessment`')

    # Training parameters
    parser.add_argument('--exp_id', default=1, type=int,
                        help='The k-th fold used for test (default: 1)')
    parser.add_argument('--multi_gpu', default=False, type=bool)
    parser.add_argument('--gpu', default=0, type=int,
                        help='Use which GPU (default: 0)')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='Number of workers to load training data (default: 2 X num_cores = 8)')

    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch size for training (default: 4)')
    parser.add_argument('--epochs', default=60, type=int,
                        help='Number of epochs for training (default: 10000)')

    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Learning rate (default: 0.0005)')
    parser.add_argument('--lr_decay_ratio', default=0.50, type=float,
                        help='learning rate multiply lr_decay_ratio (default: 0.90)')
    parser.add_argument('--lr_decay_epoch', default=50, type=int,
                        help='Learning rate decay after lr_decay_epoch (default: 10)')

    # Choose Database
    parser.add_argument('--database', default='TID2013', type=str,
                        help="Choose one of the Databases")
    parser.add_argument('--database_path', default='/media/shuyuej/Projects/Dataset/', type=str,
                        help='Database Path (default: ./dataset/)')
    parser.add_argument('--save_model_path', default='./save_model/', type=str,
                        help='Choose to save the model or not (default: ./save_model/)')

    # CNN Hyper-parameters
    parser.add_argument('--n_patches_train', default=1, type=int,
                        help='Number of patches for CNN model (default: 1)')
    parser.add_argument('--patch_size', default=224, type=int,
                        help='Image Patch Size for CNN model (default: 224)')

    # Others
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay (default: 5 X 10^-4)')

    args = parser.parse_args()

    # Get database info
    args.data_info, args.im_dir = database(benchmark=args.database, path=args.database_path)
    args.num_dis_type = distortion_type(database=args.database)

    run(args)
