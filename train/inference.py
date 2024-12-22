import torch
import torch.nn as nn
from torchvision import transforms
from dataset import SegmentationDataset
from utils import CombinedLoss
from Unet import SimpleUNet
from UnetPlusPlus import UNetPlusPlus
from UnetPlusPlusImproved import UNetPlusPlusImproved
from train import inference
import os
import shutil


def create_output_subdirs(input_dir, output_dir):
    """
    Create corresponding subdirectories in the output directory
    based on the structure of the input directory.
    """
    for root, dirs, files in os.walk(input_dir):
        for subdir in dirs:
            input_subdir = os.path.join(root, subdir)
            relative_path = os.path.relpath(input_subdir, input_dir)
            output_subdir = os.path.join(output_dir, relative_path)
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)


def process_subfolders(model, input_dir, output_dir, transform, device, num_classes):
    """
    Perform inference on images in all subfolders of the input directory
    and save the results in the corresponding subfolders of the output directory.
    """
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, relative_path)

                # Ensure the output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Run inference and save the result
                inference(model, input_path, output_path, transform, device, num_classes)


if __name__ == "__main__":

    # Paths and device setup
    inference_image_dir = "/home/ouvic/ML/ML_Final/test_2d_z/"
    output_dir = "/home/ouvic/ML/ML_Final/test_2d_z_result_1/"

    checkpoint_path = "/home/ouvic/ML/ML_Final/code/train/test_model/SimpleUnet_1214_z_100.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data augmentation and transformation
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    num_classes = 2
    model = SimpleUNet(num_classes=num_classes).to(device)

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded model weights from {checkpoint_path}")

     # Run inference
    inference(model, inference_image_dir, output_dir, transform, device, num_classes)
