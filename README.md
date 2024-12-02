# Metal Defect Classification Using EfficientNet

This repository contains a comprehensive pipeline for image classification of manufacturing defects using EfficientNetB0. It is tailored for metal defect detection, focusing on classifying defects such as "bent," "color," "scratch," and "good" samples. The pipeline includes dataset organization, model training, evaluation, and visualization.

## Features

- **Dataset Organization**: Script to reorganize datasets for proper training and testing splits.
- **Deep Learning Model**: EfficientNetB0 model with fine-tuned classification layers.
- **Data Augmentation**: Extensive augmentation techniques including random rotations, flips, and color jittering.
- **Evaluation Metrics**: Accuracy, confusion matrix, ROC curves, and classification report.
- **Cross-Validation**: Grid search with k-fold cross-validation for hyperparameter optimization.
- **Visualization**: Sample predictions, training history, class distributions, and ROC curves.

## Dataset

This project uses a metal defect dataset containing classes: `bent`, `color`, `scratch`, and `good`. Each class is split into training and testing subsets with an 80:20 ratio.

### Dataset Licensing

The dataset is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. For more details, refer to the `readme.txt` and `license.txt` files in the `model/metal_nut/` directory.

## Prerequisites

- Python 3.8+
- PyTorch
- torchvision
- NumPy
- scikit-learn
- matplotlib
- seaborn
- PIL

## Results

The pipeline provides:

- **Accuracy Metrics**: Training, validation, and testing accuracies.
- **Visualization Outputs**: Confusion matrix, ROC curves, Class distribution plots, Training history (accuracy and loss), Sample predictions.
