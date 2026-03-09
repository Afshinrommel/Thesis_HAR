In this study, the model was initially trained using the UCI Human Activity Recognition (UCI HAR) dataset, employing a UCILSTM-CNN architecture. However, the initial performance was relatively limited, particularly considering the primary objective of enabling the model to accurately recognize activities when the smartphone is placed in the user’s pocket. To address this limitation, the pre-trained model was subsequently fine-tuned using the MotionSense dataset, which contains sensor data collected directly from smartphones positioned in users’ pockets. This fine-tuning stage led to a noticeable improvement in model performance (WITHOUTAUG).

In the next stage, a statistical analysis of the MotionSense dataset was conducted to examine the distribution of activity classes. The analysis revealed that the stair-related activities were significantly underrepresented compared to the walking class, resulting in a class imbalance problem. To mitigate this issue, a targeted data augmentation strategy was applied specifically to the stair-related classes in order to balance them with the walking class.

After applying the augmentation and balancing procedure, the model was trained again (WITHAUG), which further improved its classification performance. The final trained model was then converted into the .mlpackage format to enable deployment on the iPhone platform using Apple’s Core ML framework.

It should also be noted that the gravity direction within the coordinate axes differs between the UCI HAR dataset and the MotionSense dataset. Consequently, prior to the fine-tuning stage, coordinate axis alignment (axis swapping) was performed to ensure consistency between the datasets. This preprocessing step was applied in both experimental settings, with and without data augmentation, to maintain comparable input representations across datasets.


Hybrid CNN-LSTM Human Activity Recognition
This repository contains a deep learning pipeline for Human Activity Recognition (HAR) using a hybrid architecture that combines Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks with an integrated Attention Mechanism. The project focuses on processing multi-channel inertial sensor data (accelerometer and gyroscope) to classify human movement patterns with high precision.

Project Overview
The core of this project is a sophisticated deep learning model designed to extract both spatial features and temporal dependencies from wearable sensor signals. It utilizes the UCI HAR and MotionSense datasets for training and is validated against the WISDM dataset to ensure cross-dataset robustness.

Key Technical Highlights
Architecture: Hybrid CNN-LSTM + Attention Layer.

Sensor Inputs: 9-channel data including body acceleration (x, y, z), angular velocity (x, y, z), and total acceleration (x, y, z).

Optimization: Implements accurate physical axis alignment to ensure consistency between different sensor orientations.

Deployment: Optimized for mobile environments via CoreML conversion with support for FP16 quantization for the Apple Neural Engine (ANE).

Testing: Automated parity testing infrastructure to verify model accuracy between Python training and on-device inference.

Model Architecture
The model follows a sequential feature extraction pipeline:

1D Convolutional Layers: Extract local spatial correlations and patterns within the sensor channels.

Max Pooling & Batch Normalization: Reduce dimensionality and ensure training stability.

LSTM Layer: Captures long-term temporal dependencies and the "oscillatory" nature of human gait.

Attention Mechanism: Learns to weight the most significant temporal segments within a motion window.

Softmax Output: Predicts one of 6 activity classes (Walking, Jogging, Sitting, Standing, etc.).

Dataset Details
The pipeline handles three major benchmarks:

UCI HAR: Initial training set consisting of 6 basic Activities of Daily Living (ADL).

MotionSense: Used for fine-tuning, involving smartphone-based data from 24 participants performing 6 activities.

WISDM: Used for cross-dataset validation to measure model generalization in "unseen" environments.

Workflow
Preprocessing: Signals are partitioned into temporal windows of 128 samples (2.56 seconds at 50Hz) with a 50% overlap.

Normalization: Data is scaled using MinMaxScaler within a [-1, 1] range to preserve signal integrity for Tanh/Relu activations.

Training: Implements learning rate schedulers and early stopping to prevent over-fitting.

Conversion: The Keras model is converted into a .mlpackage for iOS deployment.

Parity JSON Generation: Exports sample predictions and expected probabilities to a JSON format for verifying mobile implementation accuracy.

Requirements
Python 3.x

TensorFlow / Keras

CoreMLTools

Scikit-learn, Pandas, NumPy

Matplotlib & Seaborn (for evaluation)

Usage
Run Training: Execute lstm_cnn_architecture.ipynb to train the model on the UCI or MotionSense dataset.

Evaluate: The notebook generates learning curves and normalized confusion matrices automatically.

Export: Locate the generated best_keras_model.h5 and converted CoreML models in the designated output directories.