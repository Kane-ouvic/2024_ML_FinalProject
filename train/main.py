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
from train import train_model, inference
from random import seed
import segmentation_models_pytorch as smp
import os

if __name__ == "__main__":

    # training path
    image_dir = "/home/ouvic/ML/ML_Final/dataset_2d_x/train/imgs"
    mask_dir = "/home/ouvic/ML/ML_Final/dataset_2d_x/train/masks"
    
    # validation path
    val_image_dir = "/home/ouvic/ML/ML_Final/dataset_2d_x_test/valid/imgs"
    val_mask_dir = "/home/ouvic/ML/ML_Final/dataset_2d_x_test/valid/masks"
    
    # inference path
    inference_image_dir = "/home/ouvic/ML/ML_HW2/dataset/test/imgs"  
    output_dir = "/home/ouvic/ML/ML_HW2/predict2/pred_UnetPlusPlus_1212"  # save inference results
    
    # checkpoint_path = "/home/ouvic/ML/ML_HW2/project/pth/UnetPlusPlusImproved_1211.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data augmentation and transformation
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    augment_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 512)),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    
    # Hyperparameters
    num_classes = 2
    num_epochs = 100
    learning_rate = 1e-4
    batch_size = 32

    # Load data
    train_loader, val_loader = prepare_dataloaders(
        image_dir, mask_dir, val_image_dir, val_mask_dir, batch_size=batch_size, 
        augment_factor=0, transform=transform, augment_transform=augment_transform
    )
    torch.cuda.manual_seed(42)
    
    # Model, loss function and optimizer
    # model = SimpleUNet(num_classes=num_classes).to(device)
    # model = UNet_2048_2(num_classes=num_classes).to(device)
    # model = UNetPlusPlus(num_classes=num_classes, encoder_name='resnext101_64x4d').to(device)
    model = UNetPlusPlusImproved(num_classes=num_classes, encoder_name='resnext101_64x4d').to(device)
    criterion = CombinedLoss2(weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    
    # if os.path.exists(checkpoint_path):
    #     model.load_state_dict(torch.load(checkpoint_path))
    #     print(f"Loaded model weights from {checkpoint_path}")

    # Train model
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)
    # Execute inference
    inference(model, inference_image_dir, output_dir, transform, device, num_classes)

