import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Specifying the directory where the .npy prepared data files are located
prepared_data_dir = '/home/trg1/alok/model/prepareddata/'

# Loading the training and testing data
train_data = np.load(os.path.join(prepared_data_dir,'train_data.npy'))
test_data = np.load(os.path.join(prepared_data_dir,'test_data.npy'))

# Assuming the data is in the form [input_data, target_data]
x_train, y_train = train_data[:, 0], train_data[:, 1]
x_test, y_test = test_data[:, 0], test_data[:, 1]

# Add a time dimension to data
x_train = np.expand_dims(x_train, axis=1)  # Now shape is (samples, 1, rows, cols)
x_test = np.expand_dims(x_test, axis=1)  # Now shape is (samples, 1, rows, cols)

# Get the input shape from the training data
input_shape = x_train.shape[1:]

# Defining the model
model = Sequential()
model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), return_sequences=True, activation='relu', input_shape=input_shape))
model.add(BatchNormalization())
model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), return_sequences=False, activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(1)) # Number of neurons in the output layer

# Compiling the model
opt = Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='mse') 

# Callbacks Functions
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)
earlystop = EarlyStopping(monitor='val_loss', patience=5)

# Training the model
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test), callbacks=[checkpoint, earlystop])  
