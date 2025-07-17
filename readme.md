#  Crop Disease Classification with CNN

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.8+-blue)
![Status](https://img.shields.io/badge/status-Active-brightgreen)
![Model](https://img.shields.io/badge/model-CNN-yellow)

---

##  Project Overview

Crop diseases are a leading cause of reduced agricultural productivity worldwide. This project uses a Convolutional Neural Network (CNN) to classify different types of crop diseases from leaf images. By leveraging deep learning, we aim to support farmers and agricultural experts in accurately and efficiently diagnosing diseases using image-based recognition.

---

##  Objectives

- Build a robust image classification model using CNN.
- Process and augment leaf image data for better generalization.
- Evaluate model performance and visualize predictions.
- Deploy a user-friendly system for detecting crop diseases from new images.

---

##  Dataset

The dataset used contains high-resolution images of crop leaves, categorized by disease type.

Sample Dataset Grid:

<img width="1002" height="966" alt="f7ac3e07-d024-40a0-bd81-f20fab26010a" src="https://github.com/user-attachments/assets/6de8b10f-1fc3-408c-bc6f-5f0b43715ea7" />

Each class contains:
- Diseased leaf images (e.g., blight, rust, mosaic)
- Healthy leaf images

---

##  Model Architecture

The model is a custom CNN built using TensorFlow/Keras. Key features include:
- Multiple convolutional and pooling layers
- ReLU activations and dropout regularization
- Flattened fully connected output layer with softmax

---

##  Methodology

1. Data Loading and Preprocessing
2. Data Augmentation (flips, zooms, shifts)
3. CNN Model Building
4. Model Training & Evaluation
5. Visual Interpretation of Predictions

---

##  Visualizations

### 1. Accuracy and Loss Curve

<img width="1291" height="526" alt="94a430ad-d40a-478a-90a5-476fb74dea82" src="https://github.com/user-attachments/assets/55c9fc70-3494-4105-8c2e-0e17473b5525" />

Shows training and validation accuracy/loss over epochs.

---

### 2. Confusion Matrix

<img width="833" height="782" alt="cf24c38f-efae-491c-83aa-fcfd1d33c806" src="https://github.com/user-attachments/assets/b1a93473-18c9-48fd-8e28-0a0f25acb8fa" />

Gives insight into true vs. predicted classifications.

---

### 3. Disease Class Distribution

<img width="821" height="453" alt="cd0ee282-0e6a-4f07-a2ef-3730f75ad206" src="https://github.com/user-attachments/assets/2f1d03f5-8815-4bdf-b0d0-52dc2eaecc55" />

---

##  Results

- Final Training Accuracy: ~97.96%
- Validation Accuracy: ~91.59%

---

## Challenges:
Similar symptoms between diseases

Variability in farmer descriptions

## Future work:
Expand dataset
Use transfer learning (e.g., EfficientNet)

Multilingual NLP models

Integration with real-time advisory systems

---

## How to Run

### Prerequisites
Ensure you have Python 3.8+ installed on your system.

### Step 1: Clone the Repository
```bash
git clone https://github.com/DianaMayalo/Crop-Disease-Detection
cd Crop-Disease-Detection
```

### Step 2: Install Required Dependencies
```bash
pip install streamlit tensorflow numpy pillow scikit-learn joblib
```

### Step 3: Verify Required Model Files
Ensure these files are in your project directory:
- `cnn_model.keras` (CNN model for image classification)
- `mnb_nlp_pipeline.pkl` (Multinomial Naive Bayes pipeline for text classification)
- `pesticide_dict.pkl` (Pesticide recommendation dictionary)
- `streamlit_app.py` (Main application file)

### Step 4: Launch the Streamlit Application
```bash
streamlit run streamlit_app.py
```

### Step 5: Access the Application
The application will open in your default web browser at `http://localhost:8501`

### Using the Application

**Image Classification (CNN Tab):**
1. Click on the "📷 Image Upload (CNN)" tab
2. Upload a crop leaf image (JPG, PNG, or JPEG format)
3. The CNN model will predict the disease and provide pesticide recommendations

**Text Description (NLP Tab):**
1. Click on the "✏️ Text Description (NLP)" tab
2. Enter a description of crop symptoms (e.g., "yellow spots on leaves")
3. The NLP model will classify the disease and suggest appropriate treatments
