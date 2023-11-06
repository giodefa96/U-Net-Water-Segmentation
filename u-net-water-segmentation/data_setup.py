import os

import numpy as np
from PIL import Image
import albumentations as A

import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split

import hyperparameters as hp


class WaterBodiesDataset(Dataset):  
    """Water Bodies dataset."""  
    def __init__(self, img_dir, mask_dir, transform=None):  
        self.img_dir = img_dir  
        self.mask_dir = mask_dir  
        self.image_files = os.listdir(img_dir)  
        self.transform = A.Compose([  
            A.Resize(hp.Hyperparameters.HEIGHT,hp.Hyperparameters.WIDTH),  
            A.HorizontalFlip(),  
            A.RandomBrightnessContrast(p=0.5),  
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),  
        ])  
  
    def __len__(self):  
        return len(self.image_files)  
  
    def __getitem__(self, idx):  
        img_path = os.path.join(self.img_dir, self.image_files[idx])  
        mask_path = os.path.join(self.mask_dir, self.image_files[idx])  
        image = Image.open(img_path)
        mask = Image.open(mask_path).convert('L')   
          
        img,mask=np.array(image),np.array(mask)  
        if self.transform is not None:    
            augmentations = self.transform(image=img, mask=mask)    
            img = augmentations["image"]    
            mask = augmentations["mask"]  
          
        # Convert image from numpy array to tensor  
        img = np.transpose(img, (2, 0, 1))  
        img = img/255.0  
        
        img = torch.tensor(img).float()
          
        mask = np.expand_dims(mask, axis=0)  
        mask = mask/255.0  
        # torch.Size([32, 1, 128, 128]) remove the 1 dim
        mask = torch.tensor(mask).float()
        mask = mask.squeeze(0)
        mask = torch.where(mask > 0.5, torch.tensor(1), torch.tensor(0))  # Convert mask to binary  
          
        # Turn on gradient for image  
        img = img.detach().clone().requires_grad_(True)  
  
        return {'image': img, 'mask': mask}   


def create_dataloader(
    dataset_path: str,
    mask_path: str,
    len_split: float,
    batch_size: int,
    num_workers: int
    ):
    
    waterbody_dataset = WaterBodiesDataset(dataset_path, 
                             mask_path,
                             transform=None)
    
    # Determine the lengths of the splits
    train_len = int(len_split * len(waterbody_dataset))  # 80% for training example
    test_len = len(waterbody_dataset) - train_len  # 20% for testing example

    # Create the train and test datasets
    train_dataset, test_dataset = random_split(waterbody_dataset, [train_len, test_len])
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=num_workers,
                                                   pin_memory=True,)
    val_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=num_workers,
                                                 pin_memory=True)
    
    return train_dataloader, val_dataloader