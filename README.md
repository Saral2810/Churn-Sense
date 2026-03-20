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
2️⃣ Install Dependencies
pip install streamlit pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn shap
🖥️ Usage
▶️ Run Interactive Dashboard
streamlit run App.py

⚙️ Steps:

Upload your dataset (CSV)

Enter:

Customer Lifetime Value (CLV)

Retention Cost

Success Rate

Download the generated Action List (SEND OFFER customers)

📊 Deep-Dive Analysis

Open the notebook:

Customer Churn Code.ipynb
Includes:

⚖️ SMOTE Data Balancing

💸 Financial Confusion Matrix (Money Matrix)

🔎 SHAP Explainability

📈 Model Evaluation

💰 Methodology
🧮 Profit Optimization Engine
Net Profit = (True Positives × CLV × Success Rate)
             - (Predicted Positives × Retention Cost)

The system evaluates thresholds from 0 → 1 to identify the point of maximum ROI.
🎯 Strategic Segmentation

💎 High Value / High Tenure (VIPs)
→ Immediate intervention (calls, premium offers)

📉 Low Value / Low Tenure
→ Automated, low-cost campaigns

📈 Visual Outputs

📊 Profit Curve → Best ROI threshold

📌 Feature Importance → Key churn drivers

💸 Money Matrix → Financial impact of predictions

👨‍💻 Contributor

Saral Singhal
