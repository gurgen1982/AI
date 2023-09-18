# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 15:16:49 2023

@author: user
"""

import numpy as np


from keras.models import Sequential
from keras.layers import LSTM, Dense

import matplotlib.pyplot as plt



# Assuming you have your cryptocurrency price data in a numpy array called 'price_data'
# Preprocess the data and create input sequences with corresponding labels
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

# Normalize the data (scaling it to a range of 0 to 1)
def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# Assuming 'price_data' is a 1D numpy array of cryptocurrency prices
normalized_data = normalize_data(price_data)
sequence_length = 10  # Number of past price movements to use as input features
X_train, y_train = create_sequences(normalized_data, sequence_length)


# Build the LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, activation='relu', return_sequences=True))
    model.add(LSTM(64, activation='relu'))
    model.add(Dense(1))  # Output layer with a single output (next price change)
    model.compile(optimizer='adam', loss='mse')  # Mean squared error loss for regression task
    return model

input_shape = (sequence_length, 1)  # Shape of input data: (sequence_length, 1)
model = build_lstm_model(input_shape)


# Assuming you have a separate testing dataset 'X_test' and 'y_test'

# Train the LSTM model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))


# Evaluate the model on the test data
loss = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')

# Plot training and validation loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



# Assuming you have the latest real-time rate data as 'latest_data'
# Preprocess the latest_data and reshape it to match the input_shape
latest_data_normalized = normalize_data(latest_data)
latest_data_input = latest_data_normalized[-sequence_length:].reshape(1, sequence_length, 1)

# Make predictions using the trained model
predicted_change = model.predict(latest_data_input)
predicted_price = (latest_data[-1] + predicted_change[0][0]) * (np.max(price_data) - np.min(price_data)) + np.min(price_data)
print(f'Predicted price: {predicted_price}')
