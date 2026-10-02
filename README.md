# Another-Step-Toward-Success
This is my First DL project.  Name-Daksh Chaudhary CU Id-CU26220204 B.TECH AI/ML Sec-c
Project name - NeuroDigit-DL
# 🔢 NeuroDigit: Deep Learning Handwritten Character Recognition

## 📌 Project Context & Overview
Developed as a B.Tech CSAIML Semester 1 project, this repository implements a Multilayer Perceptron (MLP) Artificial Neural Network framework to recognize and classify handwritten numerical characters (digits 0 through 9). Mastering handwritten character analysis represents the definitive baseline for Computer Vision engineering pipelines used across modern AI industries.

## 🛠️ Software Stack
- **Deep Learning Framework:** TensorFlow / Keras (Sequential API)
- **Language Stack:** Python 3

## 📊 Dataset Foundation
The engine utilizes the built-in **MNIST Dataset**:
- **Inputs:** 28x28 pixel grayscale matrices containing handwritten digits.
- **Pre-processing:** Dimensional flattening and pixel normalisation values scaled between 0.0 and 1.0.
- **Output Classes:** 10 structural nodes tracking probabilities for digits 0-9.

## 🧠 Neural Network Layer Architecture
- **Layer 1 (Input Flattening):** Converts the 2D image matrix into a 1D vector (784 features).
- **Layer 2 (Dense Hidden Layer):** Deploys 32 node connections implementing a non-linear `ReLU` activation function for shape mapping.
- **Layer 3 (Dense Output Layer):** Implements a `Softmax` activation function to compute definitive percentage classification metrics across the 10 target classes.
