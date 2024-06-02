import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import ConvLSTM2D, Conv2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import cv2

# Specifying the directory where the .npy prepared data files are located
prepared_data_dir = '/home/trg1/alok/model/PreparedData'

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
history = model.fit(train_gen, epochs=8, validation_data=test_gen, callbacks=[checkpoint, earlystop], batch_size=1)

# Debugging: Print training history to monitor loss
print(history.history)

# Load the best saved model
best_model = tf.keras.models.load_model('best_model.keras')

# Define the directory to save the predicted images
predicted_images_dir = '/home/trg1/alok/model/'

# Define a function to generate predictions
def generate_predictions(model, input_sequences):
    # Make predictions
    predictions = model.predict(input_sequences)
    # Rescale predictions to the range 0-255
    predictions = (predictions * 255).astype(np.uint8)
    return predictions

# Example usage: generate predictions for test data
predicted_images = generate_predictions(best_model, x_test)

# Save the predicted images
for i, pred_image in enumerate(predicted_images):
    cv2.imwrite(os.path.join(predicted_images_dir, f'predicted_image_{i}.png'), pred_image)
