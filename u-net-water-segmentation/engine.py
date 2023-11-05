import torch
import numpy as np
import pandas as pd

from tqdm import tqdm

import hyperparameters as hp


# Define a function to calculate the Dice score
def f1_dice_score(preds, true_mask):
    '''
    https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
    preds should be (B, 1, H, W)
    true_mask should be (B, H, W)
    '''

    f1_batch = []

    for i in range(len(preds)):
    
        f1_image = []
        img  = preds[i].to(hp.Hyperparameters.DEVICE)
        mask = true_mask[i].to(hp.Hyperparameters.DEVICE)
        
        # Change shape of img from [2, H, W] to [H, W]
        img  = torch.argmax(img, dim=0)
        img = torch.where(img > 0.5, torch.tensor(1).to(hp.Hyperparameters.DEVICE), torch.tensor(0).to(hp.Hyperparameters.DEVICE))
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
def accuracy(preds, true_mask):
    '''
    preds should be (B, 3, H, W)
    true_mask should be (B, H, W)
    '''
    accuracy_batch = []

    for i in range(len(preds)):
        img  = preds[i].to(hp.Hyperparameters.DEVICE)
        mask = true_mask[i].to(hp.Hyperparameters.DEVICE)
        
        # Change shape of img from [25, H, W] to [H, W]
        img  = torch.argmax(img, dim=0)
        img = torch.where(img > 0.5, torch.tensor(1).to(hp.Hyperparameters.DEVICE), torch.tensor(0).to(hp.Hyperparameters.DEVICE))
        
        accuracy_batch.append(torch.sum(img == mask).item() / (hp.Hyperparameters.HEIGHT*hp.Hyperparameters.WIDTH))  # FIX LATER
        
    return np.mean(accuracy_batch)


def train(model,
          train_dataloader,
          val_dataloader,
          loss_fn,
          optimizer,
          epochs,
          starting_epoch,
          device):
    
    min_val_f1 = 0.3
    
    for epoch in tqdm(range(starting_epoch+1, starting_epoch+epochs+1)):
        # Train model
        model.train()
        train_losses = []
        train_accuracy = []
        train_f1 = []
        total_train_losses  = []
        total_train_accuracy = []
        total_train_f1 = []
      
        for i, batch in enumerate(train_dataloader):
            #Extract data, labels
            # print shapes torch.Size([32, 3, 128, 128]) torch.Size([32, 1, 128, 128])
            img_batch, mask_batch = batch['image'], batch['mask']   #img [B,3,H,W], mask[B,H,W]
            img_batch = img_batch.to(device)
            mask_batch = mask_batch.to(device)
            #Train model
            optimizer.zero_grad()
            # with torch.cuda.amp.autocast():
            output = model(img_batch) # output: [B, 25, H, W]
            loss   = loss_fn(output, mask_batch)
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 6)
            optimizer.step()

            #Add current loss to temporary list (after 1 epoch take avg of all batch losses)
            f1 = f1_dice_score(output, mask_batch)
            acc = accuracy(output, mask_batch)
            train_losses.append(loss.item())
            train_accuracy.append(acc)
            train_f1.append(f1)
            print(f'Train Epoch: {epoch}, batch: {i} | Batch metrics | loss: {loss.item():.4f}, f1: {f1:.3f}, accuracy: {acc:.3f}', flush=True)
           
        # Update global metrics
        print(f'TRAIN       Epoch: {epoch} | Epoch metrics | loss: {np.mean(train_losses):.4f}, f1: {np.mean(train_f1):.3f}, accuracy: {np.mean(train_accuracy):.3f}', flush=True)      
        total_train_losses.append(np.mean(train_losses))
        total_train_accuracy.append(np.mean(train_accuracy))
        total_train_f1.append(np.mean(train_f1))
        result_train_dict = {'train_loss': total_train_losses, 'train_f1':total_train_f1, 'train_accuracy': total_train_accuracy}
      
      
        # Validate model
        model.eval()
        val_losses   = []
        val_accuracy = []
        val_f1       = []
        total_val_losses  = []
        total_val_accuracy = []
        total_val_f1 = []
      
        for i, batch in enumerate(val_dataloader):
            #Extract data, labels
            img_batch, mask_batch = batch['image'], batch['mask']
            img_batch = img_batch.to(device)
            mask_batch = mask_batch.to(device)

            #Validate model
            with torch.cuda.amp.autocast():
                output = model(img_batch)
                loss   = loss_fn(output, mask_batch)

            #Add current loss to temporary list (after 1 epoch take avg of all batch losses)
            f1 = f1_dice_score(output, mask_batch)
            acc = accuracy(output, mask_batch)
            val_losses.append(loss.item())
            val_accuracy.append(acc)
            val_f1.append(f1)
            
            print(f'Val Epoch: {epoch}, batch: {i} | Batch metrics | loss: {loss.item():.4f}, f1: {f1:.3f}, accuracy: {acc:.3f}', flush=True)
        
        # Update global metrics
        print(f'VALIDATION  Epoch: {epoch} | Epoch metrics | loss: {np.mean(val_losses):.4f}, f1: {np.mean(val_f1):.3f}, accuracy: {np.mean(val_accuracy):.3f}', flush=True)
        print('---------------------------------------------------------------------------------')
        total_val_losses.append(np.mean(val_losses))
        total_val_accuracy.append(np.mean(val_accuracy))
        total_val_f1.append(np.mean(val_f1))
        result_val_dict = {'val_loss': total_val_losses, 'val_f1':total_val_f1, 'val_accuracy': total_val_accuracy}
      
      
        # Save the model
        torch.save(model.state_dict(), f'/Users/giovannidefeudis/Documents/Code/U-Net-Water-Segmentation/chekpoints/models/PSPNet_res101_368_{epoch}.pt')
        min_val_f1 = np.mean(val_f1)

        # Save the results so far
        temp_df = pd.DataFrame(list(zip(total_train_losses, total_val_losses, total_train_f1, total_val_f1,
                                    total_train_accuracy, total_val_accuracy)),
                            columns = ['train_loss', 'val_loss', 'train_f1', 'test_f1', 'train_accuracy',
                                        'test_accuracy'])
        temp_df.to_csv('train_val_measures.csv')
        

    return result_train_dict, result_val_dict