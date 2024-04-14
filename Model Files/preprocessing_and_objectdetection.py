import cProfile
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
import cv2
import matplotlib.cm as cm
from matplotlib.colors import BoundaryNorm

# Directory containing the trimmed images
dataset_dir = '/home/trg1/alok/model/TrimmedImages'

# Directory to save the segmented images
segmented_dir = '/home/trg1/alok/model/SegmentedImages'

# Ensuring that output directories exist
os.makedirs(segmented_dir, exist_ok=True)

def main():
    # Iterate over all images in the dataset directory
    for filename in os.listdir(dataset_dir):
        # Check if the file is an image
        if filename.endswith('.png') or filename.endswith('.jpg'):
            # Constructing the full file path
            filepath = os.path.join(dataset_dir, filename)

            # Load the image
            img = cv2.imread(filepath, 0)

            # Applying median filter
            median_filtered_img = cv2.medianBlur(img, ksize=3)

            # Normalizing the image
            normalized_img = cv2.normalize(median_filtered_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

            # Reshape the image to be a list of pixel intensities
            pixels = normalized_img.reshape(-1, 1)

            adaptive_thresh_img = cv2.adaptiveThreshold(normalized_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

            # Performing K-means clustering (Object / Feature Detection)
            kmeans = KMeans(n_clusters=100, random_state=0).fit(pixels)  # Increased number of clusters

            # Replace each pixel with its cluster center
            segmented_img = kmeans.cluster_centers_[kmeans.labels_].reshape(normalized_img.shape).astype(np.uint8)

            # Apply colormap
            colormap = cm.get_cmap('tab10', 256)  # Use 'tab10' or any other qualitative colormap
            colored_img = colormap(segmented_img)

            # Convert the colored image to a format that can be used with OpenCV
            colored_img = (colored_img * 255).astype(np.uint8)

            # Convert the OpenCV image (RGBA) to a matplotlib image (RGB)
            colored_img_rgb = cv2.cvtColor(colored_img, cv2.COLOR_RGBA2RGB)

            # Resize the image to match the original image size
            original_img = cv2.imread(filepath)
            original_height, original_width = original_img.shape[:2]
            resized_img = cv2.resize(colored_img_rgb, (original_width, original_height), interpolation = cv2.INTER_LINEAR)

            # Create a new matplotlib figure and set its size
            plt.figure(figsize=(10, 10))

            # Display the image
            plt.imshow(resized_img)

            # Define boundaries for your colorbar
            boundaries = np.linspace(0, 255, 11) # Adjust this as per your requirement

            # Create a BoundaryNorm object with given boundaries and colormap
            norm = BoundaryNorm(boundaries, colormap.N)

            # Add a discrete colorbar
            plt.colorbar(ticks=boundaries, norm=norm)

            # Add a title
            plt.title('IR Satellite Image Feature Detection')

            # Save the figure
            plt.savefig(os.path.join(segmented_dir, 'segmented_' + filename))

            # Close the figure to free up memory
            plt.close()

# Run the profiler
cProfile.run('main()')
