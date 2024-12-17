# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 15:57:43 2024

@author: user
"""
import torch
import torch.nn as nn

# Loss fn
class DiceLoss2D(nn.Module):
    def __init__(self, num_classes):
        super(DiceLoss2D, self).__init__()
        self.num_classes = num_classes
    
    def forward(self, preds, targets):
        smooth = 1e-6
        targets = targets.squeeze(1)
        targets_onehot = nn.functional.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        intersection = (preds * targets_onehot).sum(dim=(2, 3))
        pred_sum, target_sum = preds.sum(dim=(2, 3)), targets_onehot.sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)
        cls_dice_loss =  1.0 - dice
        
        return cls_dice_loss.mean()

class FocalLoss2D(nn.Module):
    def __init__(self, num_classes, gamma=2.0):
        super(FocalLoss2D, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
    
    def forward(self, preds, targets):
        smooth = 1e-6
        targets = targets.squeeze(1)
        targets_onehot = nn.functional.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        p_t = (preds * targets_onehot).sum(dim=1)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = - focal_weight * torch.log(p_t + smooth)
        
        return focal_loss.mean()
    
class VarFocalLoss2D(nn.Module):
    def __init__(self, num_classes, gamma=2.0):
        super(VarFocalLoss2D, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
    
    def forward(self, preds, targets):
        smooth = 1e-6
        targets = targets.squeeze(1)
        targets_onehot = nn.functional.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        correct_pixel = (preds.argmax(dim=1) == targets.squeeze(1))
        incorrect_pixel = ~correct_pixel
        
        p_t = (preds * targets_onehot).sum(dim=1)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = - focal_weight * torch.log(p_t + smooth)
        correct_focal_loss = (correct_pixel.float() * focal_loss).sum(dim=(1,2)) / (correct_pixel.sum(dim=(1,2)) + smooth)
        incorrect_focal_loss = (incorrect_pixel.float() * focal_loss).sum(dim=(1,2)) / (incorrect_pixel.sum(dim=(1,2)) + smooth)
        var_focal_loss = correct_focal_loss + incorrect_focal_loss
        
        return var_focal_loss.mean()
    
class CombinedLoss(nn.Module):
    def __init__(self, num_classes, weight=[0.5, 0.5]):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss2D(num_classes)
        self.varfocal_loss = VarFocalLoss2D(num_classes, gamma=2.0)
        self.weight = weight
    
    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        varfocal = self.varfocal_loss(inputs, targets)
        return self.weight[0] * dice + self.weight[1] * varfocal