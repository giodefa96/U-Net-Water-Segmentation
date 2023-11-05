import matplotlib.pyplot as plt
import torch

from PIL import Image  
import numpy as np  

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
    


# def plot_true_pred(preds):
#     for idx in range(100):
#     # plot two images
#     fig, ax = plt.subplots(1, 3, figsize=(12, 6))
#     ax[0].imshow(res[idx][1].detach().numpy(), cmap='gray')
#     ax[0].set_title('pred')

#     ax[1].imshow(list_data[idx].detach().numpy().transpose(1,2,0), cmap='gray')
#     ax[1].set_title('Mask')

#     ax[2].imshow(list_mask[idx].detach().numpy(), cmap='gray')
#     ax[2].set_title('Mask')


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