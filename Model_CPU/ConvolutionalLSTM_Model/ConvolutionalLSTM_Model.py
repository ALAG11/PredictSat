import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import ConvLSTM2D, Conv2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

# Specifying the directory where the .npy prepared data files are located
prepared_data_dir = 'FINAL PREPARED DATASET DIRECTORY PATH'

# Loading the training and testing data
train_data = np.load(os.path.join(prepared_data_dir, 'train_data.npy'))
test_data = np.load(os.path.join(prepared_data_dir, 'test_data.npy'))

# Splitting the data into input and target data
x_train, y_train = train_data[:, :-1], train_data[:, -1]
x_test, y_test = test_data[:, :-1], test_data[:, -1]

# Reshape the input data
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 512, 512, 3)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 512, 512, 3)

# Get the input shape from the training data
input_shape = x_train.shape[1:]

# Defining the model
input_layer = Input(shape=input_shape)
model = Sequential()
model.add(ConvLSTM2D(filters=16, kernel_size=(3, 3), return_sequences=True, activation='relu', padding='same'))
model.add(ConvLSTM2D(filters=16, kernel_size=(3, 3), return_sequences=False, activation='relu', padding='same'))
model.add(Conv2D(filters=3, kernel_size=(3, 3), padding='same', activation='sigmoid'))  # Output layer with 3 filters

# Assign the input layer to the model
model = Model(inputs=input_layer, outputs=model(input_layer))

# Compiling the model
opt = Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='mse')

# Callbacks Functions
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', mode='min', save_best_only=True)
earlystop = EarlyStopping(monitor='val_loss', patience=5)

# Custom data generator
class CustomDataGen(tf.keras.utils.Sequence):
    def __init__(self, x_data, y_data, batch_size):
        self.x_data = x_data
        self.y_data = y_data
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x_data) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x_data[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y_data[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array(batch_x), np.array(batch_y)

# Using the custom data generator
train_gen = CustomDataGen(x_train, y_train, batch_size=1)
test_gen = CustomDataGen(x_test, y_test, batch_size=1)

# Training the model
history = model.fit(train_gen, epochs=10, validation_data=test_gen, callbacks=[checkpoint, earlystop], batch_size=1)

# Debugging: Print training history to monitor loss
print(history.history)

# Load the best saved model
best_model = tf.keras.models.load_model('best_model.keras')

# Define a function to generate predictions for the next forecasted image
def generate_forecasted_image(model, input_sequence):
    prediction = model.predict(input_sequence)  # Predict the next image
    # Rescale prediction to 0-255
    forecasted_image = (prediction * 255).astype(np.uint8)
    return forecasted_image

# Define the directory to save the predicted and side-by-side images
output_dir = 'PREDICTED IMAGE OUTPUT DIRECTORY PATH'

# Example usage: generate forecasted image for a specific test sample
index = 0
forecasted_image = generate_forecasted_image(best_model, x_test[index:index+1])

# Rescale the actual image for comparison
y_test_rescaled = (y_test * 255).astype(np.uint8)
actual_image = y_test_rescaled[index]

# Convert both images from RGB to BGR for cv2 saving (cv2 expects BGR)
actual_image_bgr = cv2.cvtColor(actual_image, cv2.COLOR_RGB2BGR)
forecasted_image_bgr = cv2.cvtColor(forecasted_image[0], cv2.COLOR_RGB2BGR)

# Calculate the evaluation metrics between the predicted and actual image
def evaluate_predictions(y_true, y_pred):
    mse_value = np.mean((y_true - y_pred) ** 2)
    ssim_value = ssim(y_true, y_pred, multichannel=True)
    psnr_value = psnr(y_true, y_pred, data_range=255)
    return mse_value, ssim_value, psnr_value

# Calculate the evaluation metrics for the first test sample (example)
mse_value, ssim_value, psnr_value = evaluate_predictions(actual_image, forecasted_image[0])

# Create a side-by-side comparison of actual and forecasted images
def save_side_by_side_image(original, predicted, file_path):
    # Concatenate images horizontally
    combined_image = np.concatenate((original, predicted), axis=1)
    # Save the combined image to the file path
    cv2.imwrite(file_path, combined_image)

# Save the side-by-side image
output_file = os.path.join(output_dir, 'side_by_side_comparison.png')
save_side_by_side_image(actual_image_bgr, forecasted_image_bgr, output_file)

# Display the images and evaluation metrics (display expects RGB format)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(actual_image)  # Original is in RGB
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(forecasted_image[0])  # Forecasted is in RGB
axes[1].set_title(f'Predicted Image\nMSE: {mse_value:.2f}, SSIM: {ssim_value:.4f}, PSNR: {psnr_value:.2f}')
axes[1].axis('off')

plt.tight_layout()
plt.show()