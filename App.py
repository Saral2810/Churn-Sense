import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
import shap

# --- PAGE CONFIG ---
st.set_page_config(page_title="Churn Prediction & Retention HQ", layout="wide")
st.title("📊 Customer Retention Command Center")
st.markdown("Use AI to predict churn, segment customers, and optimize retention budgets.")

# --- SIDEBAR: BUSINESS INPUTS ---
st.sidebar.header("⚙️ Strategy Settings")
uploaded_file = st.sidebar.file_uploader("Upload Customer Data (CSV)", type="csv")
st.sidebar.markdown("---")
CLV = st.sidebar.number_input("Customer Lifetime Value ($)", value=500, step=50)
RETENTION_COST = st.sidebar.number_input("Cost of Retention Offer ($)", value=50, step=10)
SUCCESS_RATE = st.sidebar.slider("Campaign Success Probability", 0.1, 1.0, 0.5)

# --- 1. LOAD & TRAIN (Cached for Speed) ---
@st.cache_data
def load_data(file):
    if file is not None:
        df = pd.read_csv(file)
    else:
        # Default fallback if no file uploaded yet
        try:
            df = pd.read_csv('customer_churn_dataset-training-master.csv')
        except:
            st.error("Please upload a dataset to proceed.")
            return None, None, None, None, None
            
    # Basic Cleaning
    if 'CustomerID' in df.columns:
        df = df.drop('CustomerID', axis=1)
    df = df.dropna()
    
    # Encoding
    categorical_cols = ['Gender', 'Subscription Type', 'Contract Length']
    existing_cat = [c for c in categorical_cols if c in df.columns]
    df_encoded = pd.get_dummies(df, columns=existing_cat, drop_first=True)
    
    X = df_encoded.drop('Churn', axis=1)
    y = df_encoded['Churn']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test, X

@st.cache_resource
def train_model(X_train, y_train):
    # SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_res)
    
    # Models
    clf_lr = LogisticRegression(max_iter=1000)
    clf_rf = RandomForestClassifier(n_estimators=50)
    clf_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    
    ensemble = VotingClassifier(
        estimators=[('lr', clf_lr), ('rf', clf_rf), ('xgb', clf_xgb)],
        voting='soft'
    )
    ensemble.fit(X_train_scaled, y_res)
    
    return ensemble, scaler

# Execute Logic
X_train, X_test, y_train, y_test, X_raw = load_data(uploaded_file)

if X_train is not None:
    with st.spinner('Training AI Model (with SMOTE balancing)...'):
        model, scaler = train_model(X_train, y_train)
        X_test_scaled = scaler.transform(X_test)
        y_probs = model.predict_proba(X_test_scaled)[:, 1]

    # --- 2. PROFIT OPTIMIZATION ---
    def calculate_profit(threshold):
        pred = (y_probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        revenue = tp * CLV * SUCCESS_RATE
        spend = (tp + fp) * RETENTION_COST
        return revenue - spend

    thresholds = np.arange(0, 1, 0.05)
    profits = [calculate_profit(t) for t in thresholds]
    optimal_idx = np.argmax(profits)
    optimal_threshold = thresholds[optimal_idx]
    max_profit = profits[optimal_idx]

    # --- DASHBOARD METRICS ---
    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Projected Savings", f"${max_profit:,.0f}")
    col2.metric("🎯 Optimal Risk Threshold", f"{optimal_threshold:.2f}")
    col3.metric("👥 Customers to Target", f"{(y_probs >= optimal_threshold).sum()}")

    # --- TABS FOR ANALYSIS ---
    tab1, tab2, tab3 = st.tabs(["📈 Profit Curve", "🧩 Strategic Clusters", "📋 Action List"])

    with tab1:
        st.subheader("Profit Optimization Analysis")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(thresholds, profits, color='green', linewidth=2)
        ax.axvline(optimal_threshold, color='red', linestyle='--')
        ax.set_title("Profit vs. Intervention Threshold")
        ax.set_xlabel("Risk Threshold")
        ax.set_ylabel("Net Profit ($)")
        st.pyplot(fig)
        st.caption(f"The Peak of the curve is at {optimal_threshold:.2f}. Targeting customers above this risk score yields maximum ROI.")

    with tab2:
        st.subheader("AI Customer Segmentation")
        
        # Filter At-Risk Customers
        churn_indices = np.where(y_probs >= optimal_threshold)[0]
        
        if len(churn_indices) > 0:
            target_customers = X_test.iloc[churn_indices].copy()
            
            # Clustering (K-Means)
            cols_for_cluster = ['Tenure', 'Monthly Bill'] # Ensure these exist in your CSV
            try:
                kmeans = KMeans(n_clusters=4, random_state=42)
                target_customers['Cluster'] = kmeans.fit_predict(target_customers[cols_for_cluster])
                
                # Plot
                fig2, ax2 = plt.subplots(figsize=(10, 5))
                sns.scatterplot(
                    data=target_customers, x='Tenure', y='Monthly Bill', 
                    hue='Cluster', palette='viridis', s=100, ax=ax2
                )
                ax2.set_title("Segmentation of At-Risk Customers")
                st.pyplot(fig2)
                
                # Logic Guide
                st.info("""
                **Strategy Guide:**
                * **High Bill / High Tenure (VIPs):** Call immediately. Offer deep discount.
                * **Low Bill / Low Tenure:** Send automated email. Low priority.
                """)
            except Exception as e:
                st.warning(f"Could not perform clustering. Check column names. Error: {e}")
        else:
            st.write("No customers meet the risk threshold to analyze.")

    with tab3:
        st.subheader("Downloadable Retention List")
        results = X_test.copy()
        results['Risk_Score'] = y_probs
        results['Recommended_Action'] = results['Risk_Score'].apply(
            lambda x: "SEND OFFER" if x >= optimal_threshold else "DO NOTHING"
        )
        
        # Show High Risk only
        high_risk_df = results[results['Recommended_Action'] == "SEND OFFER"].sort_values('Risk_Score', ascending=False)
        st.dataframe(high_risk_df.head(10))
        
        csv = high_risk_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Target List (CSV)",
            csv,
            "retention_targets.csv",
            "text/csv",
            key='download-csv'
        )

# --- 4. EXPLAINABILITY (Sidebar) ---
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Diagnostics")
if st.sidebar.button("Show Feature Importance"):
    # Extract RF model
    rf = model.named_estimators_['rf']
    importances = pd.Series(rf.feature_importances_, index=X_raw.columns)
    
    st.subheader("What drives churn?")
    fig3, ax3 = plt.subplots()
    importances.nlargest(10).plot(kind='barh', color='#4c72b0', ax=ax3)
    ax3.set_title("Top 10 Churn Drivers")
    st.pyplot(fig3)