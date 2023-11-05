import os  
import zipfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi  
  
# Initialize the Kaggle API  
api = KaggleApi()  
api.authenticate(api_key='80bb9faaeb88ca37cd2619db14886bbf')  # You can also provide your API key as a string here
  
# Define the dataset you want to download  
# Format: [username]/[dataset-name]  
dataset = "franciscoescobar/satellite-images-of-water-bodies"  
  
# Define the path where you want to download the dataset  
download_path = "data/water_body"  
  
# Check if the download path exists, if not, create it  
if not os.path.exists(download_path):  
    os.makedirs(download_path)  
  
# Download the dataset  
api.dataset_download_files(dataset, path=download_path, unzip=True)  
  
print("Dataset downloaded successfully!")  


# Unzip pizza, steak, sushi data
with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
    print("Unzipping pizza, steak, sushi data...") 
    zip_ref.extractall(image_path)

# Remove zip file
os.remove(data_path / "pizza_steak_sushi.zip")