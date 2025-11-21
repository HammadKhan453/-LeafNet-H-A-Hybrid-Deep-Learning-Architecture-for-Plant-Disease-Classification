🌿 LeafNet-H: A Hybrid Deep Learning Architecture for Plant Disease Classification

LeafNet-H is a custom-designed hybrid Convolutional Neural Network (CNN) built for high-accuracy plant leaf disease detection.
The model is optimized for real-world agricultural datasets, supporting GPU acceleration, automated preprocessing, and high-performance learning from image-based plant leaf samples.

This repository provides:

✔ Complete model architecture

✔ Data loading + preprocessing pipeline

✔ Visualizations and evaluation metrics

✔ Training logs, callbacks, and results

📌 Problem Background

Plant diseases severely impact agricultural productivity worldwide. Accurate early detection is crucial.
Previous solutions include:

1. Classical ML Approaches

SVM, Random Forest, KNN, Logistic Regression

Require manual feature extraction

Perform poorly on raw images

Limited generalization to real-world conditions

2. Traditional CNNs

Models like:

Simple 3–4 layer CNNs

LeNet-5 variants

Custom shallow networks

Limitations:

Cannot capture complex multiscale leaf textures

Struggle with background noise, illumination changes

Overfit easily on small/medium plant disease datasets

3. Transfer Learning Models

ResNet50, VGG16, MobileNetV2, EfficientNet

Limitations:

Very large model size

High inference cost, unsuitable for edge IoT devices

Often require heavy fine-tuning

Domain mismatch → suboptimal for leaf disease patterns

🚀 Introducing LeafNet-H (Hybrid CNN)

LeafNet-H is designed specifically for plant leaf disease classification, unlike generic CNN architectures.

It combines:

Deep feature extraction (via hierarchical convolutions)

Efficient lightweight modules for fast training

Hybrid blocks that mix wide and deep feature maps

Skip connections to improve gradient flow

Regularization-heavy design to eliminate overfitting

The model is created to balance:

🟢 High accuracy

🟢 Robust generalization

🟢 Moderate model size

🟢 Faster inference (mobile/IoT friendly)

🏗️ LeafNet-H Architecture

Based on the notebook, the model contains:

🔹 Input Layer
Input(shape = (224, 224, 3))


Standardized for plant leaf image datasets.

🔹 Block 1 – Shallow Feature Extraction

Captures basic textures, edges, and color gradients.

Conv2D(32, 3×3) → ReLU  
MaxPooling2D  
BatchNormalization  

🔹 Block 2 – Deep Convolutional Feature Maps

Captures leaf structure patterns.

Conv2D(64, 3×3)  
Conv2D(64, 3×3)  
MaxPooling2D  
Dropout

🔹 Block 3 – Hybrid Wide–Deep Fusion

This is the core innovation.

LeafNet-H uses wide filters and deep stacked layers in parallel:

Branch A (Wide): Conv2D(128, wide filters)
Branch B (Deep): Conv2D(64) → Conv2D(64)
Fusion: Concatenate()


Purpose:

Wide branch → captures large-scale leaf disease patterns

Deep branch → focuses on fine-grained textures

🔹 Block 4 – High-Level Semantic Extraction
Conv2D(256)  
Conv2D(128)  
MaxPooling2D  
BatchNormalization

🔹 Dense Layers

Flatten features and classify:

Flatten  
Dense(256) → ReLU  
Dropout(0.5)
Dense(128) → ReLU
Dense(num_classes, activation="softmax")

🌟 Why LeafNet-H Is Better
Challenge	Previous Methods	LeafNet-H Solution
Fine-grained disease patterns	Struggle with texture variation	Hybrid CNN captures multi-scale features
Overfitting	Common in custom CNNs	Dropout + BN + hybrid structure reduce overfitting
Heavy models	Transfer learning is huge	LeafNet-H is lightweight and fast
Real-world noise	Traditional CNNs fail	Hybrid feature fusion boosts robustness
Limited dataset	Models overfit	Regularization + data pipeline improves results

Result: LeafNet-H achieves high accuracy while remaining optimized for deployment.

🧪 Pipeline in the Notebook
1. Dataset Loading

Uses image_dataset_from_directory with:

Automated labeling

80–20 train-validation split

On-the-fly preprocessing

2. Visualization

Random batch preview

Class distribution plots

Training curve graphs

Confusion matrix

Per-class probabilities

3. Training Enhancements

ModelCheckpoint

ReduceLROnPlateau

EarlyStopping (accuracy-based)

4. Evaluation

Accuracy, loss curves

Confusion matrix

Per-class analysis

Misclassified samples
