# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from unittest.mock import patch
import torch

import numpy as np
import cv2

from utils.transforms import transform_preds

def get_max_preds(cfg, patch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, patch_size])
    '''
    assert isinstance(patch_heatmaps, np.ndarray), \
        'patch_heatmaps should be numpy.ndarray'
    assert patch_heatmaps.ndim == 3, 'patch_images should be 4-ndim'

    batch_size = patch_heatmaps.shape[0] # -1
    num_joints = patch_heatmaps.shape[1] # 17
    patch_size = patch_heatmaps.shape[2] # 25

    patch_height = cfg.MODEL.PATCH_SIZE[0] //2 # [8]
    patch_width = cfg.MODEL.PATCH_SIZE[1] //2
    width = (cfg.MODEL.IMAGE_SIZE[0] // cfg.MODEL.PATCH_SIZE[1] + 1) * 2 # [34]
    patch_index = np.arange(patch_size).astype(np.float32)
    preds = np.zeros((batch_size, num_joints, 2)).astype(np.float32)
    # patch_preds = patch_heatmaps# np.softmax(patch_heatmaps, axis = 2)
    patch_center = 0.5 * np.tile([patch_width -1 , patch_height -1], (patch_size, 1)) # [25, 2]
    width_index = (patch_index % width * patch_width + patch_center[:, 0]) # [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, ....]
    height_index = (np.floor(patch_index / width) * patch_height + patch_center[:, 1])
    pad_size = np.tile([patch_width // 2, patch_height // 2], (batch_size, num_joints, 1)) # [4, 4]
    
    x = (width_index * patch_heatmaps).sum(axis = 2).astype(np.float32) - pad_size[:, :, 0] * 2
    y = (height_index * patch_heatmaps).sum(axis = 2).astype(np.float32) - pad_size[:, :, 1] * 2
    preds[:, :, 0] = np.clip(x, 0, cfg.MODEL.IMAGE_SIZE[0])
    preds[:, :, 1] = np.clip(y, 0, cfg.MODEL.IMAGE_SIZE[1])
    # print(preds)
    maxvals = np.amax(patch_heatmaps, 2).reshape((batch_size, num_joints, 1))
    # print(maxvals)
    # pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    # pred_mask = pred_mask.astype(np.float32)

    return preds, maxvals
# def get_max_preds(cfg, patch_heatmaps):
#     '''
#     get predictions from score maps
#     heatmaps: numpy.ndarray([batch_size, num_joints, patch_size])
#     '''
#     assert isinstance(patch_heatmaps, np.ndarray), \
#         'patch_heatmaps should be numpy.ndarray'
#     assert patch_heatmaps.ndim == 3, 'patch_images should be 4-ndim'

#     batch_size = patch_heatmaps.shape[0] # -1
#     num_joints = patch_heatmaps.shape[1] # 17
#     patch_size = patch_heatmaps.shape[2] # 25

#     patch_height = cfg.MODEL.PATCH_SIZE[0] //2 # [8]
#     patch_width = cfg.MODEL.PATCH_SIZE[1] //2
#     width = (cfg.MODEL.IMAGE_SIZE[0] // cfg.MODEL.PATCH_SIZE[1] + 1) * 2 # [34]
#     patch_index = np.arange(patch_size).astype(np.float32)
#     preds = np.zeros((batch_size, num_joints, 2)).astype(np.float32)
#     # patch_preds = patch_heatmaps# np.softmax(patch_heatmaps, axis = 2)
#     patch_center = 0.5 * np.tile([patch_width -1 , patch_height -1], (patch_size, 1)) # [25, 2]
#     width_index = (patch_index % width * patch_width + patch_center[:, 0]) # [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, ....]
#     height_index = (np.floor(patch_index / width) * patch_height + patch_center[:, 1])
#     pad_size = np.tile([patch_width // 2, patch_height // 2], (batch_size, num_joints, 1)) # [4, 4]
    
#     preds[:, :, 0] = (width_index * patch_heatmaps).sum(axis = 2).astype(np.float32) - pad_size[:, :, 0] * 2
#     preds[:, :, 1] = (height_index * patch_heatmaps).sum(axis = 2).astype(np.float32) - pad_size[:, :, 1] * 2
#     # print(preds)
#     maxvals = np.amax(patch_heatmaps, 2).reshape((batch_size, num_joints, 1))
#     # print(maxvals)
#     # pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
#     # pred_mask = pred_mask.astype(np.float32)

#     return preds, maxvals



def get_final_preds(cfg, patch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(cfg, patch_heatmaps)

    # width = cfg.MODEL.IMAGE_SIZE[1]
    # height = cfg.MODEL.IMAGE_SIZE[0]

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1]]
        )

    return preds, maxvals
    