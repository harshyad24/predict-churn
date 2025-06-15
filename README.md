 🏦 Bank Churn Prediction

A Machine Learning project to predict whether a customer is likely to churn using the Customer Churn dataset from Kaggle. This project implements a Random Forest Classifier, includes hyperparameter tuning and cross-validation, and provides visual performance evaluation metrics.

---
 📌 Project Description

Customer churn refers to the loss of clients or customers. Predicting churn helps businesses understand which customers are at risk of leaving, enabling proactive retention strategies. This project uses supervised learning to build a model that predicts customer churn based on historical data.

The model is trained on the **Telco Customer Churn dataset** and employs a **Random Forest algorithm** due to its robustness, accuracy, and ability to handle feature interactions.

---
 ✨ Features

- 🔍 Exploratory Data Analysis (EDA)
- 🧼 Data Cleaning & Preprocessing
- 🧠 Random Forest Classifier for churn prediction
- 🔄 Hyperparameter tuning using GridSearchCV
- 🧪 Cross-validation for model robustness
- 📊 Confusion matrix and classification report
- 💾 Model saving with joblib
- 📈 Visuals for evaluation

---
 🛠️ Installation
 pip install -r requirements.txt

---
🚀 Usage

Run the Python script
python main.py

The Jupyter notebook
jupyter notebook notebooks/Bank_Churn_Prediction.ipynb

---
📦 Dependencies
Python 3.8+
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib

---
📊 Model Performance
| Metric    | Score |
| --------- | ----- |
| Accuracy  | 0.83  |
| Precision | 0.79  |
| Recall    | 0.74  |
| F1 Score  | 0.76  |

---
🧠 Supervised Learning Overview
This project uses supervised learning, where:
The model is trained on labeled data (features + churn outcome).
It learns patterns from the data to predict future customer churn.
It is validated with cross-validation and evaluated with accuracy metrics.

---
📚 References
📘[Customer Churn Dataset - Kaggle](https://www.kaggle.com/datasets/shubhammeshram579/bank-customer-churn-prediction)

🔍 scikit-learn Documentation (https://scikit-learn.org/stable/)

📊 Confusion Matrix Explanation (https://en.wikipedia.org/wiki/Confusion_matrix)
