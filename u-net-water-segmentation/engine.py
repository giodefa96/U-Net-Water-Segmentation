import torch
import numpy as np
import pandas as pd

from tqdm import tqdm

import hyperparameters as hp

import utils


# Define a function to calculate the Dice score
def f1_dice_score(preds, true_mask, device):
    '''
    https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
    preds should be (B, 1, H, W)
    true_mask should be (B, H, W)
    '''

    f1_batch = []
    for i in range(len(preds)):
    
        f1_image = []
        img  = preds[i].to(device)
        mask = true_mask[i].to(device)
        
        # Change shape of img from [2, H, W] to [H, W]
        img  = torch.argmax(img, dim=0)
        img = torch.where(img > 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device))
        for label in range(2):
            if torch.sum(mask == label) != 0:
                area_of_intersect = torch.sum((img == label) * (mask == label))
                area_of_img       = torch.sum(img == label)
                area_of_label     = torch.sum(mask == label)
                f1 = 2*area_of_intersect / (area_of_img + area_of_label)
                f1_image.append(f1)
        
        f1_batch.append(np.mean([tensor.cpu() for tensor in f1_image]))
    return np.mean(f1_batch)
    
    
# Accuracy
def accuracy(preds, true_mask, device):
    '''
    preds should be (B, 3, H, W)
    true_mask should be (B, H, W)
    '''
    accuracy_batch = []

    for i in range(len(preds)):
        img  = preds[i].to(device)
        mask = true_mask[i].to(device)
        
        # Change shape of img from [25, H, W] to [H, W]
        img  = torch.argmax(img, dim=0)
        img = torch.where(img > 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device))
        
        accuracy_batch.append(torch.sum(img == mask).item() / (hp.Hyperparameters.HEIGHT*hp.Hyperparameters.WIDTH))  # FIX LATER
        
    return np.mean(accuracy_batch)


def train_step(model, dataloader, loss_fn, optimizer, device):  
    model.train()  
    train_losses = []  
    train_accuracy = []  
    train_f1 = []  
  
    for i, batch in enumerate(dataloader):  
        img_batch, mask_batch = batch['image'], batch['mask']  
        img_batch = img_batch.to(device)  
        mask_batch = mask_batch.to(device)  
  
        optimizer.zero_grad()  
        output = model(img_batch)  
        loss = loss_fn(output, mask_batch)  
        loss.backward()  
        optimizer.step()  
  
        f1 = f1_dice_score(output, mask_batch, device)  
        acc = accuracy(output, mask_batch, device)  
        train_losses.append(loss.item())  
        train_accuracy.append(acc)  
        train_f1.append(f1)  
        
        print(f'batch: {i} | Batch metrics | loss: {loss.item():.4f}, f1: {f1:.3f}, accuracy: {acc:.3f}', flush=True)
  
    return np.mean(train_losses), np.mean(train_f1), np.mean(train_accuracy)  
  
  
def test_step(model, dataloader, loss_fn, device):  
    model.eval()  
    val_losses = []  
    val_accuracy = []  
    val_f1 = []  
  
    for i, batch in enumerate(dataloader):  
        img_batch, mask_batch = batch['image'], batch['mask']  
        img_batch = img_batch.to(device)  
        mask_batch = mask_batch.to(device)  
  
        with torch.no_grad():  
            output = model(img_batch)  
            loss = loss_fn(output, mask_batch)  
  
        f1 = f1_dice_score(output, mask_batch, device)  
        acc = accuracy(output, mask_batch, device)  
        val_losses.append(loss.item())  
        val_accuracy.append(acc)  
        val_f1.append(f1)  
        print(f'batch: {i} | Batch metrics | loss: {loss.item():.4f}, f1: {f1:.3f}, accuracy: {acc:.3f}', flush=True)
  
    return np.mean(val_losses), np.mean(val_f1), np.mean(val_accuracy)  



def train(model,
          train_dataloader,
          val_dataloader,
          loss_fn,
          optimizer,
          epochs,
          starting_epoch,
          device):
    
    total_train_losses  = []  
    total_train_accuracy = []  
    total_train_f1 = []  
    total_val_losses  = []  
    total_val_accuracy = []  
    total_val_f1 = [] 
    
    for epoch in tqdm(range(starting_epoch+1, starting_epoch+epochs+1)):
        print("Epoch: ", epoch, flush=True)
        # Train model
        train_loss, train_f1, train_accuracy = train_step(model, train_dataloader, loss_fn, optimizer, device)  
        # Update global metrics
        print(f'TRAIN       Epoch: {epoch} | Epoch metrics | loss: {train_loss:.4f}, f1: {train_f1:.3f}, accuracy: {train_accuracy:.3f}', flush=True)      
        total_train_losses.append(train_loss) ,total_train_accuracy.append(train_accuracy) ,total_train_f1.append(train_f1)
        result_train_dict = {'train_loss': total_train_losses, 'train_f1':total_train_f1, 'train_accuracy': total_train_accuracy}
      
        # Validate model
        val_loss, val_f1, val_accuracy = test_step(model, val_dataloader, loss_fn, device)
         # Update global metrics
        print(f'VALIDATION  Epoch: {epoch} | Epoch metrics | loss: {val_loss:.4f}, f1: {val_f1}, accuracy: {val_accuracy:.3f}', flush=True)
        print('---------------------------------------------------------------------------------')
        total_val_losses.append(val_loss) ,total_val_accuracy.append(val_accuracy) , total_val_f1.append(val_f1)
        result_val_dict = {'val_loss': total_val_losses, 'val_f1':total_val_f1, 'val_accuracy': total_val_accuracy}
      
        # Save the model
        utils.save_torch_model(model, epoch, '/Users/giovannidefeudis/Documents/Code/U-Net-Water-Segmentation/chekpoints/models', 'PSPNet_res101_368')

        # Save the results so far
        utils.save_metrics_to_csv(total_train_losses, total_val_losses, total_train_f1, total_val_f1, total_train_accuracy, total_val_accuracy, 'train_val_measures.csv')  
        
    return result_train_dict, result_val_dict