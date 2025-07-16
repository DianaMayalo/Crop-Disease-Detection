# Crop Disease Detection with Multimodal Learning

[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE) [![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/) [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)

## Project Overview

This project focuses on detecting crop diseases using a **multimodal approach** that combines image data (leaf photos) and textual symptom descriptions (as might be reported by farmers). The goal is to provide a more robust prediction pipeline that reflects real-world farm scenarios.

## Workflow Summary

1. **Data Cleaning & EDA:**

   * Removed corrupted or wrong format images (JPEG, PNG cleanup).
   * Explored class distributions and image sizes.
   * Visualized samples per class.

2. **Image Preprocessing:**

   * Resized all images to 128x128.
   * Normalized pixel values.
   * Applied data augmentation.

3. **NLP Component:**

   * Created synthetic textual symptom datasets.
   * Applied TF-IDF vectorization and trained a Logistic Regression classifier.

4. **Model Building:**

   * Trained three CNN models (baseline, frozen MobileNetV2, fine-tuned MobileNetV2).
   * Compared performance of image-only, text-only, and combined models.

5. **Model Interpretability:**

   * Used Grad-CAM to visualize areas in images influencing CNN predictions.

6. **Recommendation Engine:**

   * Added a nearest neighbor search to suggest visually similar leaf images from training data.

7. **Deployment:**

   * Built an interactive Gradio app to input an image and a symptom description, returning predicted disease plus similar examples.

## Results Summary

| Model                  | Test Accuracy  |
| ---------------------- | -------------- |
| Baseline CNN           | 75.6%          |
| Frozen MobileNetV2     | 82.5%          |
| Fine-Tuned MobileNetV2 | 82.5%          |
| Text-only (NLP)        | \~78% (varies) |
| Multimodal (CNN+NLP)   | \~85%          |

## Dataset Samples

![Sample Images](images/sample_grid.png)

## App Demo

![Gradio App Demo](images/app_demo.gif)

## How to Run

```bash
git clone https://github.com/yourusername/crop-disease-multimodal
cd crop-disease-multimodal
pip install -r requirements.txt
python app.py
```

## Future Improvements

* Use real farmer datasets for text inputs.
* Explore BERT or LSTM models for NLP.
* Deploy on Hugging Face Spaces for permanent hosting.

---
**Thank you for checking out this project!** Feel free to open issues or contribute improvements.
