"""
Builds Keras models for binary classification.
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def build_model():
    model = Sequential([
        Input(shape=(1,)),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
