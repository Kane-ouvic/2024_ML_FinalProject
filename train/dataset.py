import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Convert mask to tensor and keep it as long
        mask = torch.tensor(mask, dtype=torch.long)
        
        return image, mask

class AugmentedSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, augment_factor=3, augment_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.augment_factor = augment_factor
        self.augment_transform = augment_transform
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)

    def __len__(self):
        return len(self.images) * (1 + self.augment_factor)

    def __getitem__(self, idx):
        image_idx = idx // (1 + self.augment_factor)
        img_path = os.path.join(self.image_dir, self.images[image_idx])
        mask_path = os.path.join(self.mask_dir, self.masks[image_idx])

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)
        mask = torch.tensor(mask, dtype=torch.long)

        if idx % (1 + self.augment_factor) == 0:
            # Return original image and mask
            if self.transform:
                image = self.transform(image)
            return image, mask
        else:
            # Return augmented image and mask
            if self.augment_transform:
                augmented_image = self.augment_transform(image)
            return augmented_image, mask


def prepare_dataloaders(image_dir, mask_dir, val_image_dir, val_mask_dir, 
                                 batch_size=16, augment_factor=3, transform=None, augment_transform=None):
    train_dataset = AugmentedSegmentationDataset(
        image_dir, mask_dir, transform=transform, augment_factor=augment_factor, augment_transform=augment_transform
    )

    val_dataset = SegmentationDataset(val_image_dir, val_mask_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader