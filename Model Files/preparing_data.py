import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Directory containing segmented images
segmented_dir = '/home/trg1/alok/model/SegmentedImages'

# Directory to save the prepared data
prepared_data_dir = '/home/trg1/alok/model/PreparedData'

# Ensuring that output directory exists
os.makedirs(prepared_data_dir, exist_ok=True)

# Initializing a list to store the image data
data = []

# Iterating over all images in the segmented directory
for filename in sorted(os.listdir(segmented_dir)): 
    # Check if the file is an image
    if filename.endswith('.png') or filename.endswith('.jpg'):
        # Constructing the full file path of segmented image 
        filepath = os.path.join(segmented_dir, filename)
        
        # Loading the segmented image as a numpy array and append to data list if not empty 
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img.size > 0:
            data.append(img)

# Convert list to numpy array 
data = np.array(data)

# Creating Sequences 
seq_length = 10 
sequences = [data[i:i+seq_length] for i in range(data.shape[0]-seq_length)]

# Normalizing Data (0-1 range) 
normalized_sequences = [(seq - np.min(seq)) / (np.max(seq) - np.min(seq)) for seq in sequences]

# Split Data into Training and Testing Sets 
train_data, test_data = train_test_split(normalized_sequences, test_size=0.2)

# Reshape Data fit model input dimensions  
train_data = np.reshape(train_data, (len(train_data), seq_length,) + img.shape + (1,))
test_data = np.reshape(test_data, (len(test_data), seq_length,) + img.shape + (1,))

# Saving prepared data new folder (.npy files)  
np.save(os.path.join(prepared_data_dir,'train_data.npy'), train_data)
np.save(os.path.join(prepared_data_dir,'test_data.npy'), test_data)
