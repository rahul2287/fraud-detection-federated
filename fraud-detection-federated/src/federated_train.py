import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import os

# Load dataset
df = pd.read_csv("../data/synthetic_data.csv")
X = df[["amount"]].values.astype(np.float32)
y = df["is_fraud"].values.astype(np.int32)

# Simulate 3 clients
client_data = [
    (X[:2], y[:2]),
    (X[2:4], y[2:4]),
    (X[4:], y[4:])
]

def create_tf_dataset(X, y):
    return tf.data.Dataset.from_tensor_slices((X, y)).batch(2)

federated_train_data = [create_tf_dataset(*d) for d in client_data]

# Model function
def model_fn():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return tff.learning.models.from_keras_model(
        model,
        input_spec=federated_train_data[0].element_spec,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

# Federated training setup
iterative_process = tff.learning.algorithms.build_weighted_fed_avg(model_fn)
state = iterative_process.initialize()

# Training loop
for round_num in range(1, 6):
    result = iterative_process.next(state, federated_train_data)
    state = result.state
    metrics = result.metrics
    print(f"Round {round_num}, Metrics={metrics}")

# Save model from the final round
final_model = model_fn().keras_model
final_model.save("../models/federated_model_saved")
print("Model saved successfully.")
