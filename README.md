# Chest X-ray Classification with DenseNet-121

![Project Banner](https://user-images.githubusercontent.com/SHAH-MEER/project-banner.png)

---

## 🚀 Project Overview

This project focuses on building a deep learning model to classify chest X-ray images into **Normal** and **Pneumonia** categories using a fine-tuned DenseNet-121 architecture.

The model is trained using PyTorch with advanced techniques like:

- Weighted sampling to handle class imbalance
- Custom classifier head with dropout for regularization
- AdamW optimizer with weight decay
- Cosine Annealing Learning Rate Scheduler
- Early stopping to prevent overfitting

---

## 🧰 Features

- Data loaders with weighted random sampling for balanced batches
- DenseNet-121 pretrained on ImageNet, with the last dense block and classifier fine-tuned
- Training and validation loops with progress bars
- Model checkpointing based on best validation accuracy
- Confusion matrix and classification report for detailed evaluation
- Gradio web app for interactive model inference

---

## 📝 Results

- **Best Validation Accuracy:** 96.09%
- **Test Accuracy:** 88%
- **Precision / Recall / F1-Score:** Detailed in the classification report
- Confusion matrix visualization saved as an image

---

## ⚙️ Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/SHAH-MEER/chest-xray-classification.git
cd chest-xray-classification
pip install -r requirements.txt
