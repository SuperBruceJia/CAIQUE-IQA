# -*- coding: utf-8 -*-

import os
import time
import cv2
import numpy as np
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import io


def superpixel(img):
    seeds = cv2.ximgproc.createSuperpixelSEEDS(image_width=img.shape[1],
                                               image_height=img.shape[0],
                                               image_channels=img.shape[2],
                                               num_superpixels=25,
                                               num_levels=20,
                                               prior=5,
                                               histogram_bins=20,
                                               double_step=True)
    seeds.iterate(img, 10)
    labels = seeds.getLabels()
    labels[(labels == 0.)] = labels.max() + 1

    # Measure properties of labeled image regions
    regions = regionprops(labels)

    centroids = []
    for props in regions:
        cx, cy = props.centroid
        centroids.append([cx, cy])
    
    # dis_index = [128, 160, 192, 224, 256, 288, 320]
    dis_index = [224]
    # dis_index = [160, 192, 224, 256, 288]
    distances = []
    for label in range(1, labels.max() + 1):
        loc = np.where((labels == label))

        left = loc[1].min()
        top = loc[0].min()
        right = loc[1].max()
        bottom = loc[0].max()

        left_right_dis = right - left
        top_bottom_dis = bottom - top

        if left_right_dis > top_bottom_dis:
            dis = min(dis_index, key=lambda x: abs(x - left_right_dis))
        else:
            dis = min(dis_index, key=lambda x: abs(x - top_bottom_dis))

        distances.append(dis)

    return centroids, distances
