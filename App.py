import io

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

DEFAULT_DATA = 'customer_churn_dataset-training-master.csv'
ID_COL = 'CustomerID'
TARGET = 'Churn'

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
st.sidebar.markdown("---")
MODEL_TYPE = st.sidebar.radio(
    "Model", ["XGBoost (fast)", "Ensemble: LR + RF + XGBoost (slow)"],
    help="On the sample data XGBoost alone scores the same as the ensemble and trains ~10x faster."
)


# --- 1. LOAD & PREPARE ---
@st.cache_data
def load_data(file_bytes):
    df = pd.read_csv(io.BytesIO(file_bytes)) if file_bytes else pd.read_csv(DEFAULT_DATA)
    df = df.dropna()

    # Keep the ID as the index so the action list says WHICH customer to call
    if ID_COL in df.columns:
        df = df.set_index(ID_COL)

    # Encode every text column, not just a hard-coded list
    categorical_cols = df.drop(columns=TARGET).select_dtypes(exclude='number').columns.tolist()
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

    X = df_encoded.drop(columns=TARGET)
    y = df_encoded[TARGET].astype(int)

    # 60/20/20 split: the threshold is picked on "val" and reported on "test",
    # so the projected savings are not graded on the same data used to tune them
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
    return X_train, X_val, X_test, y_train, y_val, y_test


@st.cache_resource
def train_model(file_bytes, model_type):
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(file_bytes)

    # SMOTE only helps when one class is rare; the sample data is ~57/43 so it is skipped there
    if y_train.value_counts(normalize=True).min() < 0.35:
        X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

    xgb = XGBClassifier(n_estimators=300, learning_rate=0.1, eval_metric='logloss', n_jobs=-1, random_state=42)
    if model_type.startswith("XGBoost"):
        model = xgb
    else:
        # Only Logistic Regression needs scaled inputs; trees do not
        model = VotingClassifier(
            estimators=[
                ('lr', make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
                ('rf', RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)),
                ('xgb', xgb),
            ],
            voting='soft'
        )
    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]
    return model, val_probs, test_probs


# --- 2. PROFIT OPTIMIZATION (vectorized over every threshold at once) ---
def profit_curve(y_true, probs, thresholds):
    y_true = np.asarray(y_true)
    targeted = probs[None, :] >= thresholds[:, None]
    tp = (targeted & (y_true[None, :] == 1)).sum(axis=1)
    n_targeted = targeted.sum(axis=1)
    return tp * CLV * SUCCESS_RATE - n_targeted * RETENTION_COST


# Execute Logic
file_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
try:
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(file_bytes)
except FileNotFoundError:
    st.error("Please upload a dataset to proceed.")
    st.stop()

with st.spinner('Training AI Model...'):
    model, val_probs, y_probs = train_model(file_bytes, MODEL_TYPE)

thresholds = np.arange(0, 1, 0.01)
val_profits = profit_curve(y_val, val_probs, thresholds)
optimal_threshold = thresholds[np.argmax(val_profits)]

test_profits = profit_curve(y_test, y_probs, thresholds)
max_profit = profit_curve(y_test, y_probs, np.array([optimal_threshold]))[0]
mass_profit = profit_curve(y_test, y_probs, np.array([0.0]))[0]
break_even = RETENTION_COST / (CLV * SUCCESS_RATE)
y_pred = (y_probs >= optimal_threshold).astype(int)

# --- DASHBOARD METRICS ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Projected Savings", f"${max_profit:,.0f}", f"${max_profit - mass_profit:,.0f} vs. offer to everyone")
col2.metric("🎯 Optimal Risk Threshold", f"{optimal_threshold:.2f}", f"break-even risk {break_even:.2f}", delta_color="off")
col3.metric("👥 Customers to Target", f"{y_pred.sum():,} of {len(y_pred):,}")
col4.metric("🧠 Model AUC", f"{roc_auc_score(y_test, y_probs):.3f}")

if roc_auc_score(y_test, y_probs) > 0.99:
    st.warning(
        "The model is almost perfect on this data (AUC > 0.99). Real churn data is rarely this easy, "
        "so check for leaking columns or a synthetic dataset before trusting the savings figure."
    )

# --- TABS FOR ANALYSIS ---
tab1, tab2, tab3, tab4 = st.tabs(["📈 Profit Curve", "🧩 Strategic Clusters", "📋 Action List", "🔍 Model Diagnostics"])

with tab1:
    st.subheader("Profit Optimization Analysis")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(thresholds, test_profits, color='green', linewidth=2, label='Net profit (test set)')
    ax.axvline(optimal_threshold, color='red', linestyle='--', label=f'Chosen threshold ({optimal_threshold:.2f})')
    ax.axhline(0, color='black', linewidth=1)
    ax.set_title("Profit vs. Intervention Threshold")
    ax.set_xlabel("Risk Threshold")
    ax.set_ylabel("Net Profit ($)")
    ax.legend()
    st.pyplot(fig)
    st.caption(
        f"The threshold {optimal_threshold:.2f} was chosen on a separate validation set and is reported here on unseen "
        f"test customers. In theory an offer pays off whenever risk > cost / (CLV × success rate) = {break_even:.2f}."
    )

with tab2:
    st.subheader("AI Customer Segmentation")

    target_customers = X_test[y_pred == 1].copy()
    cols_for_cluster = [c for c in ['Tenure', 'Total Spend'] if c in target_customers.columns]

    if len(target_customers) >= 4 and len(cols_for_cluster) == 2:
        # Scale first so Total Spend (hundreds) does not drown out Tenure (tens)
        scaled = StandardScaler().fit_transform(target_customers[cols_for_cluster])
        target_customers['Cluster'] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(scaled)

        # Name each cluster from its average values instead of leaving bare numbers
        summary = target_customers.groupby('Cluster')[cols_for_cluster].mean()
        hi_tenure = summary['Tenure'] >= summary['Tenure'].median()
        hi_spend = summary['Total Spend'] >= summary['Total Spend'].median()
        names = np.select(
            [hi_tenure & hi_spend, ~hi_tenure & hi_spend, hi_tenure & ~hi_spend],
            ["VIP: call now", "New big spender: personal offer", "Loyal low spender: loyalty perk"],
            default="Low priority: automated email"
        )
        target_customers['Segment'] = target_customers['Cluster'].map(dict(zip(summary.index, names)))

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.scatterplot(
            data=target_customers.sample(min(len(target_customers), 5000), random_state=42),
            x='Tenure', y='Total Spend', hue='Segment', palette='viridis', s=30, ax=ax2
        )
        ax2.set_title("Segmentation of At-Risk Customers")
        st.pyplot(fig2)
        st.dataframe(
            target_customers.groupby('Segment')[cols_for_cluster].agg(['mean', 'count']).round(1)
        )
    else:
        st.write("Not enough targeted customers (or no 'Tenure' / 'Total Spend' columns) to segment.")

with tab3:
    st.subheader("Downloadable Retention List")
    results = X_test.copy()
    results['Risk_Score'] = y_probs
    # Rank by expected dollars gained from sending the offer, not by risk alone
    results['Expected_Gain'] = y_probs * CLV * SUCCESS_RATE - RETENTION_COST
    results['Recommended_Action'] = np.where(y_pred == 1, "SEND OFFER", "DO NOTHING")

    high_risk_df = results[results['Recommended_Action'] == "SEND OFFER"].sort_values('Expected_Gain', ascending=False)
    st.dataframe(high_risk_df.head(10))

    csv = high_risk_df.to_csv().encode('utf-8')
    st.download_button(
        "📥 Download Target List (CSV)",
        csv,
        "retention_targets.csv",
        "text/csv",
        key='download-csv'
    )

with tab4:
    st.subheader("How good is the model?")
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
    m2.metric("Precision (offers sent to real churners)", f"{precision_score(y_test, y_pred, zero_division=0):.3f}")
    m3.metric("Recall (churners we caught)", f"{recall_score(y_test, y_pred, zero_division=0):.3f}")

    st.subheader("What drives churn?")
    xgb_model = model if isinstance(model, XGBClassifier) else model.named_estimators_['xgb']
    importances = pd.Series(xgb_model.feature_importances_, index=X_test.columns)
    fig3, ax3 = plt.subplots()
    importances.nlargest(10).sort_values().plot(kind='barh', color='#4c72b0', ax=ax3)
    ax3.set_title("Top 10 Churn Drivers (XGBoost)")
    st.pyplot(fig3)
