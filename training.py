# File: train_lstm.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def load_data(file_path, seq_length=30):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            landmarks = list(map(float, line.strip().split(",")))
            data.append(landmarks)

    # Create sequences
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        labels.append(0)  # Replace with 1 for "fight" or custom labeling logic

    return np.array(sequences), np.array(labels)

def train_lstm_model(data_file="landmarks.txt", model_save_path="lstm_model.h5"):
    X, y = load_data(data_file)

    # Define LSTM model
    model = Sequential([
        LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
        LSTM(32),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    # Train the model
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

    # Save the model
    model.save(model_save_path)

if __name__ == "__main__":
    train_lstm_model()
