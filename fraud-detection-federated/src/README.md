# Federated Fraud Detection – Source Code Reference

This directory contains the full set of source files used in the **fraud-detection-federated** system. Each module in this directory is designed to support training, evaluation, preprocessing, and deployment of machine learning models for fraud detection in a simulated federated environment.

## 📂 Source Files Overview

### 🔁 Federated Training
- **`federated_train.py`**  
  Orchestrates federated learning across multiple clients using TensorFlow Federated. Includes model aggregation, training rounds, and evaluation on combined data.

- **`client_simulator.py`**  
  Splits the dataset and simulates local datasets for multiple federated clients. Includes support for stratified and randomized splitting.

### 🧠 Model Building & Training
- **`model_builder.py`**  
  Builds and compiles Keras models with support for simple, wide, and deep architectures. Optimizer, loss, and input shape are configurable.

- **`trainer.py`**  
  Trains a Keras model on local data. Includes callbacks like early stopping, model checkpointing, and optional visualizations of training curves.

### 📊 Evaluation & Inference
- **`evaluator.py`**  
  Evaluates model performance on test data using metrics like accuracy, precision, recall, F1-score, confusion matrix, ROC and PR curves.

- **`model_inference.py`**  
  Loads a trained model and performs inference on new transactions (manually defined or CSV-based). Supports CLI arguments and probability thresholding.

### 🧹 Data Handling
- **`data_loader.py`**  
  Loads and preprocesses raw transaction data. Includes encoding, scaling, validation, and splitting for training and testing.

- **`preprocessing.py`**  
  Applies normalization (MinMax or Standard scaling). Includes options to save/load scalers and run from CLI with CSV input/output.

### 🧪 Testing
- **`test_main.py`**  
  Runs unit tests to verify data integrity, schema compliance, and functionality of the analysis scripts.

### ⚙️ Utilities
- **`utils.py`**  
  Logging and metrics utilities. Displays and saves metrics, classification reports, and plots confusion matrices.

- **`main.py`**  
  CLI entry point for data exploration, risk analysis, and visualization. Supports commands like `--summary`, `--fraud`, `--charts`.

## 🚀 Usage Instructions

Run the main analysis:

```bash
python src/main.py --summary --fraud --charts
