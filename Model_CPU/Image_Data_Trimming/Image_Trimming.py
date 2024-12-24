from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# Directory containing the dataset
dataset_dir = 'RAW DATASET DIRECTORY PATH'

# Directory to save the trimmed images
output_dir = 'TRIMEED IMAGES DIRECTORY PATH'

# Ensuring that output directory exists
os.makedirs(output_dir, exist_ok=True)

# Iterate over all images in the dataset directory
for filename in os.listdir(dataset_dir):
    # Check if the file is an image
    if filename.endswith('.png') or filename.endswith('.jpg'):
        # Constructing the full file path
        filepath = os.path.join(dataset_dir, filename)
        
        # Opening the image file
        with Image.open(filepath) as img:
            area = (0, 305, img.width, img.height)
            
            # Crop the image
            cropped_img = img.crop(area)
            
            # Save the cropped image
            cropped_img.save(os.path.join(output_dir, filename))