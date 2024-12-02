# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 15:57:43 2024

@author: user
"""
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import random
from tqdm import tqdm
import csv
from utils import rescale_volume
from models.unet3d import UNet3D


# Data Path
image_path = os.getcwd() + r'\training_data'

# Dataset
def split_data(root_dir, train_ratio=0.8):
    pkg_list_train, pkg_list_val = [], []
    samples = sorted(os.listdir(root_dir))
    random.shuffle(samples)
    split_idx = int(len(samples) * train_ratio)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    
    for sample_name in train_samples:
        path = os.path.join(root_dir, sample_name)
        for file in os.listdir(path):
            if file.startswith("seismicCubes_") and file.endswith(".npy"):
                seismic_path = os.path.join(path, file)
            elif file.startswith("fault_") and file.endswith(".npy"):
                fault_path = os.path.join(path, file)
                
        pkg_list_train.append((seismic_path, fault_path))
        
    for sample_name in val_samples:
        path = os.path.join(root_dir, sample_name)
        for file in os.listdir(path):
            if file.startswith("seismicCubes_") and file.endswith(".npy"):
                seismic_path = os.path.join(path, file)
            elif file.startswith("fault_") and file.endswith(".npy"):
                fault_path = os.path.join(path, file)
                
        pkg_list_val.append((seismic_path, fault_path))
        
    
    return pkg_list_train, pkg_list_val

class DarkSideDataset(Dataset):
    def __init__(self, pkg_list, mode = 'train'):
        self.pkg_list = pkg_list
        self.mode = mode

    def __len__(self):
        return len(self.pkg_list)

    def __getitem__(self, idx):
        seismic_path, fault_path = self.pkg_list[idx]
        seismic_pre, fault_pre = np.load(seismic_path, allow_pickle=True), np.load(fault_path, allow_pickle=True)
        seismic_pre = rescale_volume(seismic_pre, low=2, high=98)
        seismic = torch.from_numpy(seismic_pre / 255).float()
        seismic = seismic.unsqueeze(0).permute(0, 3, 1, 2)
        fault = torch.from_numpy(fault_pre)
        fault = fault.permute(2, 0, 1)
        
        return seismic, fault

pkg_list_train, pkg_list_val = split_data(image_path)
train_dataset = DarkSideDataset(pkg_list_train)
val_dataset = DarkSideDataset(pkg_list_val)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Loss fn
def Dice_loss(predicted, target, num_classes=2, smooth=1e-6):
    target_onehot = nn.functional.one_hot(target.long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

    intersection = (predicted * target_onehot).sum(dim=(2, 3, 4))
    pred_sum, target_sum = predicted.sum(dim=(2, 3, 4)), target_onehot.sum(dim=(2, 3, 4))
    dice = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)
    dice_loss =  1.0 - dice
    
    return dice_loss.mean()

# Evaluation
def iou_score(pred, target, n_classes=2, smooth=1e-6):
    pred_onehot = nn.functional.one_hot(pred.argmax(dim=1), num_classes=n_classes).permute(0, 4, 1, 2, 3).to(torch.bool)
    target_onehot = nn.functional.one_hot(target.long(), num_classes=n_classes).permute(0, 4, 1, 2, 3).to(torch.bool)
    
    intersection = pred_onehot & target_onehot
    union = pred_onehot | target_onehot
    IoU = (intersection.sum(dim=(2, 3, 4)) + smooth) / (union.sum(dim=(2, 3, 4)) + smooth)
    
    return IoU.mean(dim=0)
    
def evaluate_model(model, loader, classes=2):
    model.eval()
    total_class_iou = torch.zeros(classes, dtype=torch.float32).cuda()
    tbar = tqdm(loader)
    with torch.no_grad():
        for images, masks in tbar:
            images = images.cuda()
            masks = masks.cuda()
            outputs = model(images)
            iou = iou_score(outputs, masks, n_classes=classes)
            total_class_iou += iou
    
    toal_mIoU = total_class_iou.mean()
    class_iou = total_class_iou / len(loader)
    return (toal_mIoU.item() / len(loader))*100, [round(x, 3) for x in class_iou.tolist()]

# Train
def train_model(model, optimizer, scheduler, train_loader, val_loader,
                num_epochs=40, classes=2, accumulation_steps=4, csv_filename='model'):
    best_miou = 0.0 # Initialize the best MIoU score
    best_weights = None  # Variable to store the best weights
    train_miou_list, val_miou_list = [], []

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        tbar = tqdm(train_loader)
        
        # Training loop
        for i, (images, masks) in enumerate(tbar):
            images = images.cuda()
            masks = masks.cuda()
            
            # model loss
            outputs = model(images)
            preds = torch.softmax(outputs, dim=1)
            loss = criterion_1(preds, masks, num_classes=classes)
            running_loss += loss.item()
            tbar.set_description('Loss: %.3f' % (loss))
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        avg_loss = running_loss / len(train_loader)

        # Evaluation loop
        train_miou, train_class_iou = evaluate_model(model, train_loader, classes=classes)
        train_miou_list.append([epoch+1] + train_class_iou + [train_miou])
        
        
        val_miou, val_class_iou = evaluate_model(model, val_loader, classes=classes)
        val_miou_list.append([epoch+1] + val_class_iou + [val_miou])
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train MIoU : {train_miou:.4f}, {train_class_iou}, Loss : {avg_loss:.4f}")
        print(f"Epoch {epoch+1}/{num_epochs}, Val   MIoU : {val_miou:.4f}, {val_class_iou}, , lr : {optimizer.param_groups[0]['lr']}")

        # Update the best model if the current model is better
        scheduler.step()
        if (val_miou > best_miou):
            best_miou = val_miou
            best_weights = model.state_dict().copy()
            
    # After all epochs, load the best model weights
    model.load_state_dict(best_weights)
    
    # Save the best model to disk
    torch.save(best_weights, '{}.pth'.format(csv_filename))
    
    # Record traing info
    header = ['Epoch'] + ['Class_{}_IoU'.format(i) for i in range(classes)] + ['mIoU']
    
    with open(csv_filename + '_train.csv', mode='a', newline='') as f:
        writer = csv.writer(f)
        
        if f.tell() == 0:
            writer.writerow(header)
            
        for epoch_iou in train_miou_list:
            writer.writerow(epoch_iou)
            
    with open(csv_filename + '_val.csv', mode='a', newline='') as f:
        writer = csv.writer(f)
        
        if f.tell() == 0:
            writer.writerow(header)
            
        for epoch_iou in val_miou_list:
            writer.writerow(epoch_iou)
    
    print(f"Training complete. Best Val MIoU: {best_miou:.4f}")

    return model  # Return the model with the best weights

# Main
if __name__ == "__main__":    
    
    model = UNet3D(in_channels=1 , num_classes= 2)
    model.cuda()
    
    criterion_1 = Dice_loss
    optimizer1 = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=10, gamma=0.5)
    
    # Stage 1
    model = train_model(model, optimizer1, scheduler1, train_loader, val_loader,
                        num_epochs=5, classes=2, accumulation_steps=4,
                        csv_filename='Model_name')