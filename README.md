# fraud-detection-federated

> A privacy-preserving fraud detection system using federated learning to analyze transaction patterns across distributed financial institutions without sharing raw data.

---

## Overview

This project introduces a novel approach to fraud detection in the financial sector using **federated learning**, allowing multiple banks or payment processors to collaboratively train AI models without exposing sensitive customer data. The system is built with real-time applicability and regulatory compliance in mind, providing a modern, secure framework for cross institutional fraud intelligence.

---

## Key Features

- Detects fraudulent patterns using federated neural networks across distributed data nodes
- Supports privacy-preserving computation without centralized data pooling
- Includes synthetic test data for evaluation and simulation
- Extensible design to integrate with real-time payment APIs

---

## Innovation & Novelty

This is one of the **first open-source prototypes** to simulate real-time fraud detection in a **federated learning** context for the financial industry. It avoids data centralization by keeping customer data **on-device** or **on-prem**, aligning with GDPR and financial compliance standards.

The architecture and algorithms were fully designed and developed by **Rahul Autade**. This work addresses a critical challenge in global fintech operations: **how to collaborate on fraud prevention without violating data privacy laws**.

---

## Impact & Adoption

-  Demonstrates a working framework for institutions to evaluate federated fraud models
-  Applicable to banks, payment gateways, and e-wallet platforms
-  Forms the basis for future AI-based fraud engines in high-risk transactional environments

---

## Author & Contributions

Created and maintained by **Rahul Autade**

- Designed model and system architecture
- Implemented federated learning algorithms using TensorFlow Federated
- Created synthetic datasets and test flows
- Wrote documentation and examples

---

## Project Structure
fraud-detection-federated/
│
├── 📁 src/                      # Core source code
│   ├── client_simulator.py      # Federated client logic (local training)
│   ├── model_builder.py         # Fraud detection model definition
│   ├── model_inference.py       # Fraud detection model interface
│   ├── main.py                  # Fraud Main funcation
│   ├── data_loader.py           # Data loading and preprocessing
│   └── utils.py                 # Utility functions (e.g., data split, logging)
│   ├── data_loader.py           # Data loading and preprocessing
│   ├── preprocessing.py         # Preprocessing
│   ├── test_main.py             # Fraud test funcation
│   └── trainer.py               # Utility functions (e.g., data split, logging)
│
├── 📁 data/                      # Data assets
│   ├── high_fraud_data.csv       # High Hit Rated Fraud Data
│   ├── low_fraud_data.csv        # Low Hit Rated Fraud Data
│   ├── merchant_pattern_data.csv # Merchant Pattern Fraud Data
│   ├── synthetic_data.csv        # Synthetic Fraud Data
│    
├── 📁 notebooks/                 # Jupyter notebooks for demo or experiments
│   └── explore_fraud_data.ipynb
│   └── federated_training_demo.ipynb
│   └── fraud_detection_demo.ipynb
│   └── model_inference_demo.ipynb
│
├── 📁 docs/                               # repository documentation
│   ├── fraud_detection_presentation.pptx  # Fraud Detection Presentation
│   └── fraud_flowchart.png                # Fraud flow flowchart
│   ├── fraud_model_metrics.png            # Fraud Model Metrics
│   └── high_fraud_data_preview.png        # High rated Fraud data preview
│   ├── industry_standards.md              # Inductry Standard Fraud detection 
│   └── low_fraud_data_preview.png         # Low rated Fraud data preview
│   ├── merchant_pattern_data_preview.png  # Mechant Patteren Fraud Data preview
│   └── project_roadmap.md                 # Project Roadmap
│   ├── project_slide_deck.md              # Project Structure
│   └── synthetic_data_preview.png         # Synthetic Data Fraud Data preview
│   └── system_architecture.png            # Fraud Detection System Architecture

│
├── requirements.txt             # Python package dependencies
├── README.md                    # Project description, usage, and documentation
├── LICENSE                      # License file (e.g., MIT, Apache 2.0)
└── .gitignore                   # Files/folders to ignore in Git tracking

