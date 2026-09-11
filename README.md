# 🌾 Rice Leaf Disease Classification

A deep learning project to automatically classify rice leaf diseases from images using CNN and transfer learning approaches.

---

## 📌 Project Overview

This project builds and compares three deep learning models to classify rice leaf diseases — Bacterial Leaf Blight, Brown Spot, and Leaf Smut — from a small dataset of 119 images. The goal was to identify which modeling approach (training from scratch vs. transfer learning) gives the most reliable results on a small agricultural image dataset.

---

## 🔧 Tech Stack

- **Language**: Python
- **Libraries**: TensorFlow/Keras, NumPy, Pandas, PIL/OpenCV, Scikit-learn, Seaborn, Matplotlib

---

## 📊 Models Compared

| Model | Approach | Performance |
|-------|----------|-------------|
| Custom CNN | Trained from scratch | Struggled — failed to classify Brown Spot and Leaf Smut |
| VGG16 | Transfer learning + fine-tuning | Significant improvement over custom CNN |
| **MobileNetV2** ✅ | Transfer learning + fine-tuning | **Best overall** — highest and most balanced Accuracy, Precision, Recall, F1-Score |

---

## 🔍 Project Workflow

1. Load and unzip dataset (119 images across 3 disease classes)
2. Sample visualization and EDA (class balance, brightness/sharpness via HSV and Laplacian variance)
3. Image preprocessing — resize to 128x128, convert to NumPy arrays
4. One-hot encoding and stratified Train/Validation/Test split (80/15/24)
5. Data augmentation (rotation, zoom, shift, flip, brightness) to counter small dataset size
6. 5-fold Stratified Cross-Validation setup and class weighting
7. Train Custom CNN, VGG16 (transfer learning + fine-tuning), and MobileNetV2 (transfer learning + fine-tuning)
8. Compare models via accuracy, precision, recall, F1-score, and K-Fold cross-validation results

---

## 🏆 Best Model Result

- **Model**: MobileNetV2 (Transfer Learning + Fine-Tuning)
- Achieved the highest and most consistent F1-score across all K-Fold cross-validation splits
- Correctly classified Bacterial Leaf Blight and Leaf Smut near-perfectly; ~80% accuracy on Brown Spot

---

## 📁 Dataset

Rice Leaf dataset (PRCP-1001-RiceLeaf) — 119 images across 3 disease classes:
- Bacterial Leaf Blight (40)
- Brown Spot (40)
- Leaf Smut (39)
