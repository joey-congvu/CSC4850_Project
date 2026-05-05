# 🚀Classification & Intelligent Spam Detection

> A machine learning project focused on **multi-dataset classification** and **real-world spam detection using NLP techniques**.

---

## 📌 Overview

This project explores how different machine learning models behave across **diverse datasets and problem types**. It is divided into two major components:

- 🔢 Multi-Class Classification (4 datasets)
- 📧 Spam Detection (Natural Language Processing)

The goal is to build a **consistent ML pipeline** while adapting to different data structures, feature spaces, and class distributions.

---

## 🧠 Key Highlights

- Built and compared multiple ML models:
  - Logistic Regression
  - Support Vector Machines (SVM)
  - K-Nearest Neighbors (KNN)
  - Random Forest
  - Decision Tree
  - Neural Network (MLP)

- Applied Stratified 5-Fold Cross-Validation for robust evaluation  
- Focused on macro precision, recall, and F1-score for imbalanced datasets  
- Used PCA for dimensionality reduction in high-dimensional datasets  
- Implemented TF-IDF vectorization for text-based spam detection  


## 🔍 Part 1: Multi-Class Classification

### 🎯 Objective
Evaluate ML models across 4 datasets with varying:
- Feature sizes (11 → 9182 features)
- Sample sizes
- Class distributions

---

### ⚙️ Pipeline

1. Handle missing values (`1e+99 → NaN`)
2. Mean imputation (train-only to avoid leakage)
3. Feature scaling (Standardization)
4. PCA (retain ~95% variance)
5. Model training + cross-validation

---

### 📊 Final Model Selection

| Dataset | Characteristics | Best Model |
|--------|----------------|-----------|
| Dataset 1 | High-dim, low samples | Linear SVM |
| Dataset 2 | Extreme high-dim | Logistic Regression |
| Dataset 3 | Nonlinear, multi-class | Random Forest |
| Dataset 4 | Low-dim, imbalanced | Random Forest |

---

### 💡 Key Insight

There is **no one-size-fits-all model** — performance depends heavily on:
- Dimensionality
- Data distribution
- Class imbalance

---

## 📧 Part 2: Spam Detection (NLP)

### 🎯 Objective
Classify emails as:
- Spam
- Ham (legitimate)

---

### 🧹 Text Preprocessing

- Lowercasing
- Remove HTML tags
- Replace URLs & emails with tokens
- Clean noise (punctuation, repeated characters)
- Combine multiple datasets

---

### 🔡 Feature Engineering

- Converted text → TF-IDF vectors
- High-dimensional sparse representation

---

### 🤖 Models Evaluated

- Decision Tree
- Neural Network (MLP)
- Linear SVM (Best)

---

### 🏆 Final Model: Linear SVM

Why it won:
- Highest F1-score
- Best spam recall (critical metric)
- Stable across folds
- Handles high-dimensional TF-IDF well

---

## 📈 Evaluation Metrics

- Accuracy (not sufficient alone)
- Macro Precision
- Macro Recall
- F1-Score ⭐

---

## ⚠️ Key Takeaways

- Macro metrics > Accuracy for imbalanced data  
- Dataset structure determines model choice  
- SVM excels in high-dimensional spaces  
- Random Forest handles nonlinear patterns well  
- Data preprocessing is critical  

---

## 🛠️ Tech Stack

- Python
- Scikit-learn
- NumPy / Pandas
- TF-IDF (NLP)
- Matplotlib

---

## 🚀 Future Improvements

- Deep learning for spam detection (LSTM / Transformer)
- Advanced hyperparameter tuning
- Feature selection optimization
- Deploy as REST API

---
## ⭐ Final Thought

**“The data decides the model — not the other way around.”**
