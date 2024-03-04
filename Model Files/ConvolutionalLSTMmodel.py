import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Specifying the directory where the .npy prepared data files are located
prepared_data_dir = '/home/trg1/alok/model/PreparedData'

# Loading the training and testing data
train_data = np.load(os.path.join(prepared_data_dir,'train_data.npy'))
test_data = np.load(os.path.join(prepared_data_dir,'test_data.npy'))

# Splitting the data into input and target data
x_train, y_train = train_data[:, :-1], train_data[:, -1]
x_test, y_test = test_data[:, :-1], test_data[:, -1]

# Add a time dimension to data
x_train = np.expand_dims(x_train, axis=2)  # Now shape is (samples, seq_length, 1, rows, cols, channels)
x_test = np.expand_dims(x_test, axis=2)  # Now shape is (samples, seq_length, 1, rows, cols, channels)

# Get the input shape from the training data
input_shape = x_train.shape[2:]

# Defining the model
model = Sequential()
model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), return_sequences=True, activation='relu', input_shape=input_shape))
model.add(BatchNormalization())
model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), return_sequences=False, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=1, kernel_size=(3, 3), padding='same', activation='sigmoid'))  # Output layer

# Compiling the model
opt = Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='mse') 

# Callbacks Functions
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)
earlystop = EarlyStopping(monitor='val_loss', patience=5)

# Training the model
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test), callbacks=[checkpoint, earlystop])  
