import torch

from PIL import Image
import numpy as np

import hyperparameters as hp

def make_predictions(model: torch.nn.Module, data: list, device: torch.device = hp.Hyperparameters.DEVICE):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # Prepare sample
            sample = torch.unsqueeze(sample, dim=0).to(hp.Hyperparameters.DEVICE) # Add an extra dimension and send sample to device

            # Forward pass (model outputs raw logit)
            pred_logit = model(sample)
            # Get prediction probability (logit -> prediction probability)
            
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0) # note: perform softmax on the "logits" dimension, not "batch" dimension (in this case we have a batch size of 1, so can perform on dim=0)
            #pred_prob  = torch.argmax(pred_logit, dim=0)
            pred_prob = torch.where(pred_prob > 0.5, torch.tensor(1).to(hp.Hyperparameters.DEVICE), torch.tensor(0).to(hp.Hyperparameters.DEVICE))
            # Get pred_prob off GPU for further calculations
            pred_probs.append(pred_prob.cpu())

        # Stack the pred_probs to turn list into a tensor
        return torch.stack(pred_probs)
    
    
def predict_flow(images, model, device):
    
    images_trans = []
    
    for image in images:
        image = Image.open(image)
        image = image.convert('RGB')
        image = image.resize((hp.Hyperparameters.WIDTH, hp.Hyperparameters.HEIGHT))
        image = np.array(image)
        image = np.transpose(image, (2, 0, 1))
        image = image/255.0
        image = torch.tensor(image).float()
        images_trans.append(image)
    
    return make_predictions(model, images_trans, device)
        
