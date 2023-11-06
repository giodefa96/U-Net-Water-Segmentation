import matplotlib.pyplot as plt

import torch

from PIL import Image  
import numpy as np  
import pandas as pd  

import hyperparameters as hp


def plot_and_save_loss(total_train_losses, total_val_losses, save_path='loss_plot.png'):
    """
    Plot and save a loss plot.

    Args:
        total_train_losses (list): List of training losses.
        total_val_losses (list): List of validation losses.
        save_path (str): Path to save the plot as an image. Default is 'loss_plot.png'.
    """
    plt.figure(figsize=(10, 8))
    plt.plot(list(range(len(total_train_losses) + 1))[1:120], total_train_losses[:120])
    plt.plot(list(range(len(total_train_losses) + 1))[1:120], total_val_losses[:120])
    # plt.plot(list(range(len(total_train_losses) + 1))[1:120], total_val_f1[:120])
    plt.legend(['train loss', 'val loss', 'val f1'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Save the plot
    plt.savefig(save_path)

    # Display the plot
    plt.show()
    

# load models
def load_model(model, model_path):
    model = model.to(hp.Hyperparameters.DEVICE)
    model.load_state_dict(torch.load(model_path))
    return model


def plot_predictions(images, preds):    
    for idx, el in enumerate(preds):    
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))    
        ax[0].imshow(el[0].detach().numpy(), cmap='gray')    
            
        # Open the image file    
        img = Image.open(images[idx])    
        # Resize the image  
        img = img.resize((hp.Hyperparameters.WIDTH, hp.Hyperparameters.HEIGHT))  
        # Convert the image data to numpy array    
        img_data = np.array(img)    
            
        ax[1].imshow(img_data, cmap='gray')    
        plt.show()    
        

def save_torch_model(model, epoch, path, model_name):
    torch.save(model.state_dict(), f'{path}/{model_name}_{epoch}.pt')



  
def save_metrics_to_csv(train_losses, val_losses, train_f1, val_f1, train_accuracy, val_accuracy, filename='train_val_measures.csv'):  
    temp_df = pd.DataFrame(list(zip(train_losses, val_losses, train_f1, val_f1, train_accuracy, val_accuracy)),  
                           columns=['train_loss', 'val_loss', 'train_f1', 'val_f1', 'train_accuracy', 'val_accuracy'])  
    temp_df.to_csv(filename, index=False)  
