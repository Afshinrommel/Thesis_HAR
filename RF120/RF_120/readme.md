MotionSense Human Activity Recognition (Random Forest)
This repository contains a machine learning pipeline for recognizing human activities using smartphone-based inertial sensor data. The project utilizes a Random Forest classifier trained on the MotionSense dataset, extracting 120 distinct features to achieve high-accuracy classification.

Project Overview
The objective of this project is to transform raw accelerometer and gyroscope data into semantically meaningful activity labels (e.g., walking, sitting, standing). The pipeline includes data ingestion, feature engineering, model training, and conversion for mobile deployment.

Key Features
Dataset: MotionSense (Smartphone-based HAR).

Feature Engineering: Extraction of 120 statistical and kinematic features from 9-channel sensor signals.

Classifier: Random Forest (RF) ensemble model.

Mobile Optimization: Conversion to CoreML format for real-time inference on iOS devices.

Pipeline Architecture
Data Preprocessing: Raw signals are segmented into temporal windows and normalized to ensure signal integrity and orientation invariance.

Feature Extraction: A vector of 120 features is generated for each window, capturing both time-domain and frequency-domain attributes.

Model Training: A Random Forest model is trained to map the 120-feature input vector to specific activity classes.

CoreML Conversion: The trained scikit-learn model is converted into a .mlpackage using coremltools, optimized for the Apple Neural Engine (ANE).

Model Metadata
Model Type: Random Forest Classifier

Input Vector Size: 120 Features

Short Description: MotionSense Random Forest (120 Features)

Output: classLabel (Predicted Activity)

Requirements
To run the notebook and reproduce the training process, you will need:

Python 3.x

Pandas

NumPy

Scikit-learn

CoreMLTools (for model conversion)

Usage
Data Preparation: Ensure the MotionSense dataset is available in the specified directory.

Training: Run the Jupyter Notebook RF_120.ipynb to process the data and train the model.

Export: The notebook will automatically save the trained model as a CoreML file (.mlpackage) for integration into mobile applications.

Deployment Note
The model is configured to use an input description of "Input Vector (Size 120)" and provides the predicted activity label as the primary output. For mobile implementation, ensure the input features are extracted in the same order as defined during the training phase.