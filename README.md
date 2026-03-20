📊 Customer Retention Command Center
An end-to-end Machine Learning solution designed to predict customer churn, segment at-risk users, and optimize retention budgets based on financial ROI.

🚀 Overview
Predicting churn is easy, but knowing which customers to save is a business challenge. This project uses an Ensemble AI model (Logistic Regression, Random Forest, and XGBoost) to calculate a "Risk Score" for every customer. It then uses a Profit Optimization Engine to find the specific risk threshold where the revenue saved outweighs the cost of intervention.

Key Features

Ensemble Modeling: Combines multiple algorithms for robust risk scoring.
SMOTE Balancing: Automatically handles imbalanced datasets to ensure the AI identifies churners effectively.
Profit Optimization Curve: Visualizes the financial "sweet spot" for marketing intervention.
Strategic Segmentation: Uses K-Means clustering to categorize churners into tiers (e.g., VIPs vs. Low Priority).
Explainable AI (SHAP): Provides a "Deep Dive" into why specific customers are leaving.
Interactive Dashboard: A Streamlit interface for business users to upload data and set financial parameters.

🛠️ Technical Stack

Language: Python 3.x
Framework: Streamlit (UI)
Data Science: Pandas, NumPy, Scikit-Learn, XGBoost, Imbalanced-Learn (SMOTE)
Visualization: Matplotlib, Seaborn, SHAP (Explainability)

📥 Installation
Clone the repository:

Bash
git clone https://github.com/yourusername/churn-retention-hq.git
cd churn-retention-hq

Install dependencies:
Bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn shap
Prepare your data:
Place your dataset (CSV) in the project folder. Ensure it contains columns for Tenure, Monthly Bill, and a target column named Churn.

🖥️ Usage
1. Running the Interactive Dashboard
The Streamlit app allows non-technical managers to run the model.
Bash
streamlit run App.py
Sidebar Settings: Input your Customer Lifetime Value (CLV), Retention Cost, and Success Rate.
Upload: Drop your customer CSV into the uploader.
Action List: Download the generated CSV of customers marked for "SEND OFFER."

2. Deep-Dive Analysis (Notebook)
Open Customer Churn Code.ipynb to view the full pipeline, including:
Statistical data balancing (SMOTE).
The Money Matrix (Financial Confusion Matrix).
SHAP force plots for individual customer diagnostics.

📈 MethodologyThe Profit EngineThe model doesn't just look for churners; it looks for profit.
{Net Profit} = ({True Positives}*{CLV}*{Success Rate}) - ({Predicted Positives}*{Retention Cost})
The system iterates through risk thresholds (0.0 to 1.0) to find where this value is highest.Strategic TiersOnce churners are identified, they are clustered into four segments:

High Value / High Tenure: The "VIP" segment requiring immediate phone calls and deep discounts.

Low Value / Low Tenure: Automated email segments with low-cost interventions.

🔍 Visualizations
The project generates several critical outputs:
The Profit Curve: A green-line graph showing the peak ROI threshold.
Feature Importance: A horizontal bar chart showing the top 10 drivers of churn (e.g., Contract type, Tenure).
Money Matrix: A heatmap showing the dollar impact of the AI's decisions.

👥 Contributors
Saral Singhal
