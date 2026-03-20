# 🚀 <span style="color:#00C4FF; font-size:40px;">ChurnSense</span>  
### 📊 Customer Retention Command Center  

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit)
![ML](https://img.shields.io/badge/Machine-Learning-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Project-Active-brightgreen?style=for-the-badge)

---

> An end-to-end Machine Learning solution to **predict customer churn, prioritize high-value users, and optimize retention strategies based on financial ROI.**

---

## 🔍 Overview

Predicting churn is easy — **knowing who to save is the real business problem.**

**ChurnSense** uses an **Ensemble AI Model** (Logistic Regression, Random Forest, XGBoost) to generate a **Risk Score** for each customer.  
It then applies a **Profit Optimization Engine** to determine the exact point where **retention cost < revenue saved**.

---

## ✨ Key Features

- 🧠 **Ensemble Modeling**  
  Combines multiple ML algorithms for robust predictions  

- ⚖️ **SMOTE Balancing**  
  Handles imbalanced datasets effectively  

- 📈 **Profit Optimization Curve**  
  Identifies the most profitable intervention threshold  

- 🎯 **Strategic Segmentation**  
  K-Means clustering divides customers into actionable tiers  

- 🔎 **Explainable AI (SHAP)**  
  Understand *why* customers churn  

- 🖥️ **Interactive Dashboard**  
  Streamlit-based UI for business users  

---

## 🛠️ Tech Stack

| Category        | Tools Used |
|----------------|-----------|
| Language       | Python 3.x |
| UI Framework   | Streamlit |
| ML Libraries   | Scikit-learn, XGBoost |
| Data Handling  | Pandas, NumPy |
| Imbalance Fix  | Imbalanced-learn (SMOTE) |
| Visualization  | Matplotlib, Seaborn |
| Explainability | SHAP |

---

## 📥 Installation

### 1️⃣ Clone Repository
```bash
git clone https://github.com/yourusername/churn-retention-hq.git
cd churn-retention-hq
```
2️⃣ Install Dependencies
```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn shap
```
🖥️ Usage
▶️ Run Interactive Dashboard
```bash
streamlit run App.py
```
---

## 🖥️ Usage Workflow

- 📂 **Seamless Data Upload**  
  Import customer dataset in CSV format directly into the dashboard  

- 💰 **Business Parameter Configuration**  
  Define key inputs:
  - Customer Lifetime Value (CLV)  
  - Retention Cost  
  - Success Rate  

- 🎯 **Actionable Decision Output**  
  Generate and download a prioritized **SEND OFFER customer list**  

---

## 📊 Deep-Dive Analysis

- 📓 **Notebook Exploration**  
  Analyze model logic using `Customer Churn Code.ipynb`  

- ⚖️ **SMOTE Balancing**  
  Corrects class imbalance for more reliable predictions  

- 💸 **Financial Confusion Matrix**  
  Evaluates predictions based on actual monetary impact  

- 🔎 **SHAP Explainability**  
  Identifies key drivers influencing churn behavior  

- 📈 **Model Evaluation**  
  Provides performance metrics and validation insights  

---

## 💰 Methodology

- 🧮 **Profit Optimization Engine**  
  Focuses on maximizing business value instead of just accuracy  

```text
Net Profit = (True Positives × CLV × Success Rate)
             - (Predicted Positives × Retention Cost)
```
💸 Money Matrix → Financial impact of predictions

👨‍💻 Contributor

Saral Singhal
