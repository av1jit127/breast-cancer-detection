# 🔬 Breast Cancer Classification & Diagnostic System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://breast-cancer-detecti0n.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-yellow.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning web application powered by **Deep Learning (Artificial Neural Networks)** to predict whether a breast tumor is **Malignant** or **Benign** based on cell nucleus characteristics.

🔗 **Live Demo:** [breast-cancer-detecti0n.streamlit.app](https://breast-cancer-detecti0n.streamlit.app/)

---

## 📌 Project Overview

Early detection of breast cancer significantly increases the chances of successful treatment. This application uses an Artificial Neural Network trained on cytological numerical features computed from fine-needle aspirate (FNA) images of breast masses.

The model standardizes raw feature inputs and classifies instances into:
- **0 - Malignant (Cancerous)**
- **1 - Benign (Non-Cancerous)**

---

## 🚀 Key Features

- **Interactive Web Interface:** Built using Streamlit for instant predictions.
- **Deep Learning Classifier:** Multi-layer feedforward neural network constructed with TensorFlow / Keras.
- **Standardized Pipeline:** Preprocessing integrated via Scikit-Learn's `StandardScaler` to ensure zero training-serving skew.
- **Preset Test Samples:** Built-in sidebar presets allowing quick one-click testing of malignant and benign samples without manual entry.
- **Model Confidence Score:** Displays softmax probability percentages along with classification results.

---

## 📊 Dataset

The model is trained on the **Wisconsin Breast Cancer Diagnostic Dataset**
