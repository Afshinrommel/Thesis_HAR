Human Activity Recognition using UCI Dataset (Random Forest → CoreML)
Overview

This project implements a Human Activity Recognition (HAR) pipeline using the UCI HAR dataset.
The workflow includes:

Data preprocessing and feature extraction

Training a Random Forest classifier

Evaluating the model performance

Converting the trained model to Apple CoreML (.mlpackage) format for deployment on iPhone

The final model can be integrated into iOS applications for on-device activity recognition.

Project Pipeline
1. Dataset

The project uses the UCI Human Activity Recognition (HAR) dataset, which contains smartphone sensor data collected from:

Accelerometer

Gyroscope

Activities included in the dataset:

Walking

Walking Upstairs

Walking Downstairs

Sitting

Standing

Laying

Sensor signals are segmented and transformed into statistical and frequency-domain features.

Feature Extraction

The notebook extracts multiple features from the raw signals, including:

Statistical Features

Mean

Standard deviation

Skewness

Kurtosis

Frequency Domain Features

FFT components

These features help capture both time-domain and frequency-domain characteristics of human movement.

Model Training

A Random Forest Classifier is used for activity classification.

Libraries

Scikit-learn

NumPy

Pandas

Training Steps

Load dataset

Extract features

Split dataset into training and testing sets

Train Random Forest model

Evaluate using standard classification metrics

Model Evaluation

The following evaluation metrics are used:

Accuracy

Confusion Matrix

Classification Report

Visualization tools used:

Matplotlib

Seaborn

These provide insights into model performance and class prediction behavior.

CoreML Conversion

After training, the model is exported to CoreML format for use in iOS applications.

Conversion Steps

Load trained .pkl model

joblib.load(model_path)

Convert using coremltools

ct.convert()

Save as:

.mlpackage
Output
uci_75_string.mlpackage

This file can be integrated directly into Xcode for real-time activity recognition on iPhone.

Installation

Install the required dependencies:

pip install scikit-learn==1.5.2 coremltools pandas numpy matplotlib scipy joblib
Project Structure
project/
│
├── UCI_75.ipynb
├── uci_only_rf_model.pkl
├── README.md
│
└── CoreML_Output/
    └── uci_75_string.mlpackage
Usage
1. Train the Model

Run the notebook:

UCI_75.ipynb

This will:

preprocess the dataset

extract features

train the Random Forest model

2. Convert Model to CoreML

The notebook includes a conversion step that generates:

.mlpackage

This model can then be used inside iOS applications.

Deployment (iOS)

To use the model in an iOS application:

Drag the .mlpackage file into Xcode

Xcode automatically generates a Swift interface

Use CoreML + CoreMotion to feed sensor data into the model

Technologies Used

Python

Scikit-learn

CoreMLTools

NumPy

Pandas

Matplotlib

Seaborn

Future Improvements

Deep learning models (CNN / LSTM / Transformer)

Real-time streaming inference

Multi-sensor fusion

Model optimization for mobile devices

If you want, I can also create a much more professional GitHub README including:

badges

architecture diagrams

pipeline illustrations

dataset explanation

model performance tables