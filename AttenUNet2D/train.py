# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 15:57:43 2024

@author: user
"""
import torch
from tqdm import tqdm
import csv
from metrics import iou_score

# Train
def train_model(model, optimizer, scheduler, train_loader, val_loader, criterion = None,
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
            loss = criterion(preds, masks)
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

# Validate   
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