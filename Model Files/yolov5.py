from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
import cv2
from yolov5.models import YOLOv5

# Load the pre-trained YOLO model
model = YOLOv5('yolov5s.pt')

# Directory containing the cropped images
dataset_dir = '/home/trg1/alok/model/croppedimages'

# Directory to save the segmented images
segmented_dir = '/home/trg1/alok/model/segmentedimages'

# Ensuring that output directories exist
os.makedirs(segmented_dir, exist_ok=True)

# Iterate over all images in the dataset directory
for filename in os.listdir(dataset_dir):
    # Check if the file is an image
    if filename.endswith('.png') or filename.endswith('.jpg'):
        # Constructing the full file path
        filepath = os.path.join(dataset_dir, filename)
        
        # Load the image
        img = cv2.imread(filepath, 0)

        # Preprocess the image for YOLO
        img = preprocess_for_yolo(img)
        
        # Perform inference with YOLO
        boxes, scores, classes = model.predict(img)
        
        # Postprocess the YOLO outputs to get a binary mask
        mask = postprocess_yolo(boxes, scores, classes)
        
        # Save the mask to disk
        cv2.imwrite(os.path.join(segmented_dir, 'segmented_' + filename), mask)
