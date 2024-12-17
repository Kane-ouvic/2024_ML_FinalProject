# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 15:57:43 2024

@author: user
"""
import os
import torch
from torch.utils.data import DataLoader
from models.network import UNet2D
from dataset import get_data, DarkSideDataset
from config import Config
from losses import DiceLoss2D, VarFocalLoss2D, CombinedLoss
from train import train_model

# Data Path
image_path = os.getcwd() + r'\training_data'

# Dataset
x_train_lst, x_val_lst = get_data(image_path, direction='x', data_type='train'), get_data(image_path, direction='x', data_type='valid')
y_train_lst, y_val_lst = get_data(image_path, direction='y', data_type='train'), get_data(image_path, direction='y', data_type='valid')
z_train_lst, z_val_lst = get_data(image_path, direction='z', data_type='train'), get_data(image_path, direction='z', data_type='valid')

x_train_set, x_val_set = DarkSideDataset(x_train_lst, direction='x'), DarkSideDataset(x_val_lst, direction='x')
y_train_set, y_val_set = DarkSideDataset(y_train_lst, direction='y'), DarkSideDataset(y_val_lst, direction='y')
z_train_set, z_val_set = DarkSideDataset(z_train_lst, direction='z'), DarkSideDataset(z_val_lst, direction='z')

x_train_loader, x_val_loader = DataLoader(x_train_set, batch_size=Config.BATCH_SIZE, shuffle=True), DataLoader(x_val_set, batch_size=Config.BATCH_SIZE, shuffle=False)
y_train_loader, y_val_loader = DataLoader(y_train_set, batch_size=Config.BATCH_SIZE, shuffle=True), DataLoader(y_val_set, batch_size=Config.BATCH_SIZE, shuffle=False)
z_train_loader, z_val_loader = DataLoader(z_train_set, batch_size=Config.BATCH_SIZE, shuffle=True), DataLoader(z_val_set, batch_size=Config.BATCH_SIZE, shuffle=False)

# Main
if __name__ == "__main__":
    
    torch.cuda.manual_seed(42)
    
    model_x = UNet2D(encoder_name=Config.ENCODER_NAME,
                     encoder_depth=Config.ENCODER_DEPTH,
                     encoder_weights=Config.PRETRAIN_WEIGHT,
                     in_channels=Config.IN_CHANNEL,
                     decoder_channels=Config.ENCODER_CHANNEL,
                     attention_type=Config.ATTENTION_TYP, # Option : none, scse
                     classes=Config.NUM_CLASSES
                     )
    
    model_y = UNet2D(encoder_name=Config.ENCODER_NAME,
                     encoder_depth=Config.ENCODER_DEPTH,
                     encoder_weights=Config.PRETRAIN_WEIGHT,
                     in_channels=Config.IN_CHANNEL,
                     decoder_channels=Config.ENCODER_CHANNEL,
                     attention_type=Config.ATTENTION_TYP, # Option : none, scse
                     classes=Config.NUM_CLASSES
                     )
    
    model_z = UNet2D(encoder_name=Config.ENCODER_NAME,
                     encoder_depth=Config.ENCODER_DEPTH,
                     encoder_weights=Config.PRETRAIN_WEIGHT,
                     in_channels=Config.IN_CHANNEL,
                     decoder_channels=Config.ENCODER_CHANNEL,
                     attention_type=Config.ATTENTION_TYP, # Option : none, scse
                     classes=Config.NUM_CLASSES
                     )
    
    loss_fn = CombinedLoss(num_classes=Config.NUM_CLASSES, weight=[0.5, 0.5])
    optimizer_x = torch.optim.Adam(model_x.parameters(), lr=Config.LEARNING_RATE)
    optimizer_y = torch.optim.Adam(model_y.parameters(), lr=Config.LEARNING_RATE)
    optimizer_z = torch.optim.Adam(model_z.parameters(), lr=Config.LEARNING_RATE)
    
    scheduler_x = torch.optim.lr_scheduler.StepLR(optimizer_x, step_size=Config.REDUCT_STEP, gamma=Config.REDUCT_FACTOR)
    scheduler_y = torch.optim.lr_scheduler.StepLR(optimizer_y, step_size=Config.REDUCT_STEP, gamma=Config.REDUCT_FACTOR)
    scheduler_z = torch.optim.lr_scheduler.StepLR(optimizer_z, step_size=Config.REDUCT_STEP, gamma=Config.REDUCT_FACTOR)
    
    # Train x
    model_x.cuda()
    model_x = train_model(model_x, optimizer_x, scheduler_x,
                          x_train_loader, x_val_loader,
                          criterion = loss_fn,
                          num_epochs=Config.NUM_EPOCHS,
                          classes=Config.NUM_CLASSES,
                          accumulation_steps=Config.ACCUMULATION_STEP,
                          csv_filename=Config.MODEL_NAME + '_x')
    del model_x
    torch.cuda.empty_cache()
    
    # Train y
    model_y.cuda()
    model_y = train_model(model_y, optimizer_y, scheduler_y,
                          y_train_loader, y_val_loader,
                          criterion = loss_fn,
                          num_epochs=Config.NUM_EPOCHS,
                          classes=Config.NUM_CLASSES,
                          accumulation_steps=Config.ACCUMULATION_STEP,
                          csv_filename=Config.MODEL_NAME + '_y')
    del model_y
    torch.cuda.empty_cache()
    
    # Train z
    model_z.cuda()
    model_z = train_model(model_z, optimizer_z, scheduler_z,
                          z_train_loader, z_val_loader,
                          criterion = loss_fn,
                          num_epochs=Config.NUM_EPOCHS,
                          classes=Config.NUM_CLASSES,
                          accumulation_steps=Config.ACCUMULATION_STEP,
                          csv_filename=Config.MODEL_NAME + '_z')
    del model_z
    torch.cuda.empty_cache()   