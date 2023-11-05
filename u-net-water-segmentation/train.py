import os
import argparse  

import torch
import data_setup, engine, model_builder, utils

import hyperparameters as hp

from torch import nn

# setup argparse  
parser = argparse.ArgumentParser(description='U-Net Water Segmentation')  
parser.add_argument('--epochs', type=int, default=hp.Hyperparameters.EPOCHS, help='Number of epochs')  
parser.add_argument('--starting_epoch', type=int, default=hp.Hyperparameters.STARTING_EPOCH, help='Starting epoch')  
args = parser.parse_args()


# setup directories
image_dir = "/Users/giovannidefeudis/Documents/Code/U-Net-Water-Segmentation/data/water_body/Images"
mask_dir  = "/Users/giovannidefeudis/Documents/Code/U-Net-Water-Segmentation/data/water_body/Masks"


# Create dataloader
train_dataloadet, test_dataloader = data_setup.create_dataloader(
    image_dir,
    mask_dir,
    hp.Hyperparameters.TRAIN_VAL_SPLIT,
    hp.Hyperparameters.BATCH_SIZE,
    num_workers=hp.Hyperparameters.NUM_WORKERS
)

model = model_builder.UNET(
    in_channels=3,
    out_channels=hp.Hyperparameters.NUM_CLASSES).to(hp.Hyperparameters.DEVICE)

# Set loss and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=hp.Hyperparameters.LR)
criterion = nn.CrossEntropyLoss().to(hp.Hyperparameters.DEVICE)

# Start Training
train_loss, val_loss = engine.train(
    model=model,
    train_dataloader=train_dataloadet,
    val_dataloader=test_dataloader,
    loss_fn=criterion,
    optimizer=optimizer,
    epochs=args.epochs,  
    starting_epoch=args.starting_epoch,  
    device=hp.Hyperparameters.DEVICE
    )

# save plot

utils.plot_and_save_loss(total_train_losses=train_loss['train_loss'], total_val_losses=val_loss['val_loss'], save_path='loss_plot.png')