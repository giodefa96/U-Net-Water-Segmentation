import predict
import model_builder
import utils

import os

import hyperparameters as hp

model = model_builder.UNET(
    in_channels=3,
    out_channels=hp.Hyperparameters.NUM_CLASSES).to(hp.Hyperparameters.DEVICE)

model = utils.load_model(model, "/Users/giovannidefeudis/Documents/Code/U-Net-Water-Segmentation/chekpoints/models/PSPNet_res101_368_2.pt")

#directory_path = '/Users/giovannidefeudis/Documents/Code/U-Net-Water-Segmentation/data/images_to_predict'  # Replace with the actual path to your directory
directory_path = '/Users/giovannidefeudis/Documents/Code/U-Net-Water-Segmentation/data/water_body/Images'  # Replace with the actual path to your directory
# List all elements (files and subdirectories) in the directory
elements = os.listdir(directory_path)
# concatenate directory path with all elements
images = [os.path.join(directory_path, el) for el in elements][0:30]

pred = predict.predict_flow(images,
                     model,
                     hp.Hyperparameters.DEVICE)

# plot predictions
utils.plot_predictions(images, pred)