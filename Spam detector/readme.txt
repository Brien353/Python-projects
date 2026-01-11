# High-Precision Spam Detection Pipeline
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/sklearn-latest-orange.svg)](https://scikit-learn.org/)

## 🚀 Project Overview
Developed a robust end-to-end Machine Learning pipeline to classify email as Spam or Ham using the Apache SpamAssassin public corpus. This project focuses on sophisticated text preprocessing and high-precision classification using Stochastic Gradient Descent (SGD).

## 🛠️ Engineering Highlights
* **Custom NLP Transformers:** Built a modular Scikit-Learn pipeline to handle raw email headers, body content, and metadata.
* **Regex Normalization:** Engineered automated feature reduction by converting URLs, Email addresses, and Numbers into generic placeholders to improve model generalization.
* **Stemming & Tokenization:** Implemented NLTK-based stemming to reduce vocabulary dimensionality.

## 📊 Performance Metrics
The model was benchmarked against a **Dummy Classifier (Baseline)** to ensure statistical significance.

| Metric | Dummy (Naive) | SGD Classifier |
| :--- | :--- | :--- |
| **Accuracy** | 74.34% | **98.99%** |
| **Precision** | 0.00% | **99.20%** |
| **Recall** | 0.00% | **96.87%** |
| **ROC-AUC** | 0.50 | **0.999** |



**Key Insight:** The 99.2% Precision ensures a near-zero False Positive rate, which is critical for user-facing email products where misclassifying "ham" is more costly than missing "spam."