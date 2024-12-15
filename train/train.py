import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_gaussian, create_pairwise_bilateral
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20, device=device, num_classes=13):
    train_loss_history = []
    val_loss_history = []
    train_iou_history = []
    val_iou_history = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_iou = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # calculate IoU
            batch_iou = calculate_iou(outputs, masks, num_classes)
            running_iou += batch_iou

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_iou = running_iou / len(train_loader)
        train_loss_history.append(epoch_train_loss)
        train_iou_history.append(epoch_train_iou)

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_iou = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_loss = criterion(outputs, masks)
                val_running_loss += val_loss.item()

                # calculate IoU
                batch_iou = calculate_iou(outputs, masks, num_classes)
                val_running_iou += batch_iou

        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_val_iou = val_running_iou / len(val_loader)
        val_loss_history.append(epoch_val_loss)
        val_iou_history.append(epoch_val_iou)
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), Config.SAVE_PATH + f'_{epoch+1}.pt')

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss}, Val Loss: {epoch_val_loss}, Train IoU: {epoch_train_iou}, Val IoU: {epoch_val_iou}")

    torch.save(model.state_dict(), Config.SAVE_PATH)
    print("Model weights saved")

    # plot Training and Validation Loss
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_loss_history, label="Training Loss")
    plt.plot(range(1, num_epochs + 1), val_loss_history, label="Validation Loss", linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.legend()
    plt.savefig('./loss.png')
    
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_iou_history, label="Training IoU")
    plt.plot(range(1, num_epochs + 1), val_iou_history, label="Validation IoU", linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title("Training and Validation IoU over Epochs")
    plt.legend()
    plt.savefig('./iou.png')
    

def inference(model, image_dir, output_dir, transform, device, num_classes):
    model.eval()  # set model to evaluation mode

    # ensure output folder exists
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():  # disable gradient calculation
        # print(os.listdir(image_dir))
        for image_name in os.listdir(image_dir):
            img_path = os.path.join(image_dir, image_name)
            image = cv2.imread(img_path)
            
            if image is None:
                print(f"Warning: Unable to read image '{img_path}'. Skipping this file.")
                continue
            
            # image transformation
            input_image = transform(image).unsqueeze(0).to(device)  # add batch dimension
            
            # model inference
            output = model(input_image)
            output = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # get the maximum value index of each pixel as class
            
            # save or display results
            output_img = Image.fromarray((output * (255 // (num_classes - 1))).astype(np.uint8))  # visualize results
            output_img.save(os.path.join(output_dir, f"{image_name}"))
            # print(f"Processed {image_name}")
            
            
def inferenceCRF(model, image_dir, output_dir, transform, device, num_classes):
    model.eval()  # set model to evaluation mode

    # ensure output folder exists
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():  # disable gradient calculation
        for image_name in os.listdir(image_dir):
            img_path = os.path.join(image_dir, image_name)
            image = cv2.imread(img_path)
            
            if image is None:
                print(f"Warning: Unable to read image '{img_path}'. Skipping this file.")
                continue
            
            # image transformation
            input_image = transform(image).unsqueeze(0).to(device)  # add batch dimension
            
            # model inference
            output = model(input_image)
            output_probs = torch.softmax(output, dim=1).squeeze(0).cpu().numpy()  # get softmax probability
            
            # build CRF
            height, width = image.shape[:2]
            d = dcrf.DenseCRF2D(width, height, num_classes)

            # set unary prediction probability
            unary = unary_from_softmax(output_probs)
            d.setUnaryEnergy(unary)

            # add pairwise parameters to keep edges
            d.addPairwiseGaussian(sxy=3, compat=3)
            d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image, compat=10)

            # perform CRF inference
            Q = d.inference(5)
            crf_output = np.argmax(Q, axis=0).reshape((height, width))

            # save or display results
            output_img = Image.fromarray((crf_output * (255 // (num_classes - 1))).astype(np.uint8))  # visualize results
            output_img.save(os.path.join(output_dir, f"{image_name}"))
            print(f"Processed {image_name} with CRF")


def calculate_iou(outputs, masks, num_classes):
    # convert model output to class prediction
    outputs = torch.argmax(outputs, dim=1)
    ious = []
    for cls in range(1, num_classes):  # assume class 0 is background
        pred = (outputs == cls)
        target = (masks == cls)
        intersection = (pred & target).sum().item()
        union = (pred | target).sum().item()
        if union == 0:
            iou = float('nan')  # if no prediction or annotation, ignore this class
        else:
            iou = intersection / union
            ious.append(iou)
    if len(ious) == 0:
        return 0.0
    else:
        return np.nanmean(ious)  # calculate mean IoU, ignore NaN

