from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import numpy as np
import os
import cv2

# Directory containing the cropped images
dataset_dir = '/home/trg1/alok/model/croppedimages'

# Directory to save the segmented images
segmented_dir = '/home/trg1/alok/model/segmentedimages'

# Ensuring that output directories exist
os.makedirs(segmented_dir, exist_ok=True)

# Load VGG16 model for feature extraction
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)

# Iterating over all images in the dataset directory
for filename in os.listdir(dataset_dir):
    # Checking if the file is an image
    if filename.endswith('.png') or filename.endswith('.jpg'):
        # Constructing the full file path
        filepath = os.path.join(dataset_dir, filename)
        
        # Loading the image
        img = cv2.imread(filepath, 0)

        # Convert grayscale image to 3-channel image
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Resize the image to 224x224 for VGG16
        img = cv2.resize(img, (224, 224))

        # Convert the image to array
        x = image.img_to_array(img)

        # Preprocess the image
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Extract features
        features = model.predict(x)

        # Save the features
        np.save(os.path.join(segmented_dir, 'features_' + filename), features)

        # Save the preprocessed image
        cv2.imwrite(os.path.join(segmented_dir, 'preprocessed_' + filename), img)
