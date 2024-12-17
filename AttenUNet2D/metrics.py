# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 15:57:43 2024

@author: user
"""
import torch
import torch.nn as nn

# Metrics
def iou_score(pred, target, n_classes=2, smooth=1e-6):
    pred_onehot = nn.functional.one_hot(pred.argmax(dim=1), num_classes=n_classes).permute(0, 3, 1, 2).to(torch.bool)
    target_onehot = nn.functional.one_hot(target.squeeze(1), num_classes=n_classes).permute(0, 3, 1, 2).to(torch.bool)
    
    intersection = pred_onehot & target_onehot
    union = pred_onehot | target_onehot
    IoU = (intersection.sum(dim=(2, 3)) + smooth) / (union.sum(dim=(2, 3)) + smooth)
    
    return IoU.mean(dim=0)