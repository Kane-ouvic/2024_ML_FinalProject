import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import SegmentationDataset, prepare_dataloaders
from utils import CombinedLoss, CombinedLoss2
from Unet import SimpleUNet, UNet_2048, UNet_2048_2 
from UnetPlusPlus import UNetPlusPlus, UNetPlusPlus2
from UnetPlusPlusImproved import UNetPlusPlusImproved
from Baseline_Unet import *
from train import train_model, inference
from random import seed
import segmentation_models_pytorch as smp
import os
from config import Config

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data augmentation and transformation
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD),
    ])
    
    augment_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD),
    ])

    # Load data
    train_loader, val_loader = prepare_dataloaders(
        Config.TRAIN_IMAGE_DIR, Config.TRAIN_MASK_DIR, 
        Config.VAL_IMAGE_DIR, Config.VAL_MASK_DIR, 
        batch_size=Config.BATCH_SIZE, 
        augment_factor=0, transform=transform, 
        augment_transform=augment_transform
    )
    torch.cuda.manual_seed(42)
    
    # Model, loss function and optimizer
        # model = SimpleUNet(num_classes=num_classes).to(device)
    # model = UNet_2048_2(num_classes=num_classes).to(device)
    # model = UNetPlusPlus(num_classes=num_classes, encoder_name='resnext101_64x4d').to(device)
    # model = UNetPlusPlusImproved(num_classes=num_classes, encoder_name='resnext101_64x4d').to(device)
    # criterion = CombinedLoss2(weight=0.5)
    # optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    
    if Config.MODEL_NAME == "SimpleUNet":
        model = SimpleUNet(num_classes=Config.NUM_CLASSES).to(device)
    elif Config.MODEL_NAME == "UNet_2048_2":
        model = UNet_2048_2(num_classes=Config.NUM_CLASSES).to(device)
    elif Config.MODEL_NAME == "UNetPlusPlus":
        model = UNetPlusPlus(num_classes=Config.NUM_CLASSES, encoder_name=Config.ENCODER_NAME).to(device)
    elif Config.MODEL_NAME == "UNetPlusPlusImproved":
        model = UNetPlusPlusImproved(num_classes=Config.NUM_CLASSES, encoder_name=Config.ENCODER_NAME).to(device)
    elif Config.MODEL_NAME == "UNetBaseline":
        model = UNetBaseline(input_channels=3, output_channels=Config.NUM_CLASSES).to(device)
    
    
    criterion = CombinedLoss2(weight=Config.LOSS_WEIGHT)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    
    
    if os.path.exists(Config.CHECKPOINT_PATH):
        model.load_state_dict(torch.load(Config.CHECKPOINT_PATH))
        print(f"Loaded model weights from {Config.CHECKPOINT_PATH}")
        
    
    # Train model
    train_model(model, train_loader, val_loader, criterion, optimizer, Config.NUM_EPOCHS)
    # Execute inference
    inference(model, Config.INFERENCE_IMAGE_DIR, Config.OUTPUT_DIR, transform, device, Config.NUM_CLASSES)

