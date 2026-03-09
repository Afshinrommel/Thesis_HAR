Hybrid Human Activity Recognition (RF-75)
This repository contains a specialized machine learning pipeline for Human Activity Recognition (HAR) using a hybrid feature set. The project focuses on a Random Forest classifier that utilizes 75 optimized features to distinguish between 7 distinct physical activities.

Project Overview
The "RF_COMBINATION_75" project is an evolution of standard HAR models, combining selected statistical and kinematic features to balance computational efficiency and predictive power. The core of the project is a Random Forest model trained to run on mobile hardware via CoreML.

Key Technical Highlights
Model Type: Random Forest Classifier (Ensemble Learning).

Feature Set: 75 Hybrid features (optimized for classification accuracy).

Classes: 7 Activity categories (e.g., walking, sitting, standing, walking upstairs/downstairs, etc.).

Deployment: Fully compatible with iOS (CoreML Framework).

Quantization: Support for weight optimization to reduce memory footprint.

Pipeline Structure
Hybrid Feature Loading: The system loads pre-calculated features derived from 9-channel inertial sensor data (Accelerometer & Gyroscope).

Model Integration: Utilizing a scikit-learn trained Random Forest model (.pkl format).

CoreML Transformation: The model is converted into a .mlpackage using coremltools, specifically targeting high performance on the Apple Neural Engine (ANE).

Metadata Configuration: includes input/output descriptions for seamless integration into Swift/Xcode projects.

Model Metadata
Input: Feature Vector (Size 75).

Output: classLabel (String representing the predicted activity).

Author:  Afshin Monfared.

Format: .mlpackage (CoreML).

Installation & Requirements
To execute the notebook and conversion script:

Bash
pip install pandas numpy scikit-learn coremltools
How to Use
Notebook Execution: Open RF_COMBINATION_75.ipynb in Google Colab or a local Jupyter environment.

Model Path: Ensure the hybrid model file (final_hybrid_rf_model_7classes.pkl) is located in your /FINAL_EXPORT/ directory.

Conversion: Run the conversion cells to generate the Hybrid_RF_75.mlpackage.

iOS Integration: Drag the resulting file into your Xcode project to automatically generate the Swift interface.