#  Crop Disease Classification with CNN

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.8+-blue)
![Status](https://img.shields.io/badge/status-Active-brightgreen)
![Model](https://img.shields.io/badge/model-CNN-yellow)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Notebook Status](https://img.shields.io/badge/Notebook-Completed-brightgreen)](./crop.ipynb)

---

##  Project Overview
This project focuses on identifying plant leaf diseases from images using Convolutional Neural Networks (CNNs). The model is trained on the publicly available PlantVillage dataset, which contains images of healthy and diseased crop leaves across 15 classes. The final solution applies data preprocessing, image augmentation, and deep learning techniques to achieve high accuracy on multi-class classification.

---

##  Objectives

- Build a robust image classification model using CNN.
- Process and augment leaf image data for better generalization.
- Evaluate model performance and visualize predictions.
- Deploy a user-friendly system for detecting crop diseases from new images.

---

##  Dataset
The dataset used contains high-resolution images of crop leaves, categorized by disease type.
- Dataset Source: [PlantVillage dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
- Classes: 15 (e.g., Pepper__bell___healthy, Potato___Early_blight, Tomato___Leaf_Mold, etc.)
- Image Size: Resized to 128x128 pixels
- Label Format: Categorical (one-hot encoded)
- Format: Folder structure by class labels

--- 

## Sample Images from the Dataset

Below is a small grid sample of the dataset classes:

| Tomato___Healthy | Potato___Late_blight | Pepper__bell___healthy | Tomato__Tomato_mosaic_virus |
|------------------|----------------------|-------------------------|-----------------------------|
| ![0d789240-9714-4378-8b63-4afb12ddfa44___RS_HL 9735](https://github.com/user-attachments/assets/39ea6144-c39b-4f28-bef9-2e0715dd5a65) | ![0e068694-63b7-4edf-a93d-f2e9f28efaa6___RS_LB 3923](https://github.com/user-attachments/assets/9bff54b9-95a9-4a81-b373-1c410f9ddb9f) | ![0a3f2927-4410-46a3-bfda-5f4769a5aaf8___JR_HL 8275](https://github.com/user-attachments/assets/f03623aa-1150-46fb-b210-e3d608adf3d0) | ![1b9dc07a-40ab-45bc-a873-1ad4212e35a3___PSU_CG 2289](https://github.com/user-attachments/assets/c1aa0cb1-24ab-4b85-ac3a-350ad968af81) |

Each class contains:
- Diseased leaf images (e.g., blight, rust, mosaic)
- Healthy leaf images

---

##  Model Architectural Approach

The model is a custom CNN built using TensorFlow/Keras. Key features include:
- Image loading via tf.keras.utils.image_dataset_from_directory
- Data augmentation using tf.keras.layers
- Model: Sequential CNN with:
  - Conv2D + MaxPooling
  - BatchNormalization
  - Dropout
  - Flatten → Dense → Softmax
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Evaluation: Accuracy and loss on validation/test set

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

### 4. Deployment Summary

To make the model accessible to real users, we deployed it through a Streamlit web application:

- The trained model was saved in .h5 format for reuse and easy loading.
- A demo web application was created using Streamlit, allowing users to upload crop leaf images and receive instant disease predictions.
- The demo showcases the model's performance and user-friendliness, simulating how it could be used in real agricultural settings.
- This proof of concept lays the groundwork for potential full-scale deployment in the future (e.g., mobile apps, IoT devices, or cloud services).

This deployment enables the practical use of our crop disease detector by farmers, researchers, and agronomists.

---

## Results

The crop disease detection model achieved promising performance, validating its effectiveness in distinguishing between healthy and diseased leaves across multiple crop types.

### Model Performance:

- The Convolutional Neural Network (CNN) achieved high accuracy on the validation and test sets:
  - Validation Accuracy: 95%+
  - Test Accuracy: ~94%
- The model showed strong generalization capabilities, with minimal overfitting due to proper use of dropout and data augmentation.
- Confusion matrix analysis indicated high true positive rates across major disease classes.

### Evaluation Metrics:

| Metric           | Score    |
|------------------|----------|
| Accuracy         | ~90%     |
| Precision        | High     |
| Recall           | High     |
| F1 Score         | Strong   |

### Sample Predictions:

- Correctly classified:
  - Potato Late Blight
- Misclassifications occurred occasionally with visually similar symptoms across crops.

### Insights:

- Deep learning is a viable solution for early disease detection in agriculture.
- Image-based disease prediction can assist farmers in decision-making and reduce crop loss.
- The trained model performs well across a diverse range of leaf types and diseases due to effective data preprocessing and augmentation.

These results reinforce the potential of AI-powered plant health monitoring systems in modern agriculture.

---

## Recommendations:
### 1. Enhance Dataset Diversity
Improve model generalization by collecting more real-world images under different lighting conditions, angles, and backgrounds. This reduces bias and improves the model's ability to perform well in practical field conditions. Also, consider balancing classes if some disease categories are underrepresented.

### 2. Integrate Explainability Tools (Grad-CAM)
Add visual interpretability features like Grad-CAM or saliency maps to highlight regions of the leaf that the model focuses on during prediction. This builds trust with agricultural experts and helps verify that the model learns the correct patterns.

### 3. Optimize the Model for Deployment
After training, convert the model using TensorFlow Lite or ONNX for deployment on mobile or edge devices. You can also apply quantization and pruning techniques to reduce model size and inference time, making it usable on devices directly used by farmers.

### 4. Create an Interactive Web Interface
Build a simple yet functional user interface using tools like Streamlit or Flask that allows users to upload a leaf image and receive a disease prediction instantly. This makes the model accessible and usable by non-technical stakeholders in the agricultural field.

---

## Challenges:
### 1. Dataset Imbalance
Some crop diseases were underrepresented in the dataset, making it difficult for the model to learn equally well across all classes. This imbalance risked overfitting on the majority classes and underperforming on rare diseases.

### 2. High Intra-Class Similarity
Several leaf diseases had very similar visual symptoms (e.g., different types of blight), making it challenging for the model to distinguish between them, especially in the absence of high-resolution features.

### 3. Variability in farmer descriptions

---

## Future work Improvements:
- Expand dataset
- Use transfer learning (e.g., EfficientNet)
- Multilingual NLP models
- Integrate Grad-CAM for explainability
- Develop a Streamlit or Flask-based frontend app
-Integration with real-time advisory systems
- Explore lightweight models for deployment on mobile devices
- Implement early stopping and learning rate scheduling

---

##  How to Run

1. Clone the repository:

https://github.com/DianaMayalo/Crop-Disease-Detection

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
1. Click on the "Image Upload (CNN)" tab
2. Upload a crop leaf image (JPG, PNG, or JPEG format)
3. The CNN model will predict the disease and provide pesticide recommendations

**Text Description (NLP Tab):**
1. Click on the "Text Description (NLP)" tab
2. Enter a description of crop symptoms (e.g., "yellow spots on leaves")
3. The NLP model will classify the disease and suggest appropriate treatments
