"""Feature definitions and the leakage-safe feature-engineering transformer.

Everything that turns a raw customer record into model input lives here so
training and serving share exactly the same code path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Raw columns as they appear in the source CSV -> snake_case API names.
RAW_TO_API = {
    "Age": "age",
    "Gender": "gender",
    "Tenure": "tenure",
    "Usage Frequency": "usage_frequency",
    "Support Calls": "support_calls",
    "Payment Delay": "payment_delay",
    "Subscription Type": "subscription_type",
    "Contract Length": "contract_length",
    "Total Spend": "total_spend",
    "Last Interaction": "last_interaction",
}

TARGET = "churn"
ID_COL = "customer_id"

RAW_NUMERIC = [
    "age",
    "tenure",
    "usage_frequency",
    "support_calls",
    "payment_delay",
    "total_spend",
    "last_interaction",
]
RAW_CATEGORICAL = ["gender", "subscription_type", "contract_length"]
RAW_FEATURES = RAW_NUMERIC + RAW_CATEGORICAL

ENGINEERED = ["spend_per_tenure", "engagement_score", "high_support_user", "payment_risk"]
MODEL_NUMERIC = RAW_NUMERIC + ENGINEERED

CATEGORIES = {
    "gender": ["Female", "Male"],
    "subscription_type": ["Basic", "Standard", "Premium"],
    "contract_length": ["Monthly", "Quarterly", "Annual"],
}

FEATURE_LABELS = {
    "age": "Age",
    "gender": "Gender",
    "tenure": "Tenure (months)",
    "usage_frequency": "Usage frequency",
    "support_calls": "Support calls",
    "payment_delay": "Payment delay (days)",
    "subscription_type": "Subscription",
    "contract_length": "Contract length",
    "total_spend": "Total spend",
    "last_interaction": "Days since last interaction",
}


def load_raw(path: str) -> pd.DataFrame:
    """Load the source CSV, normalise column names and drop unusable rows."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={**RAW_TO_API, "CustomerID": ID_COL, "Churn": TARGET})
    df = df.dropna(subset=[TARGET, *RAW_FEATURES])
    df[TARGET] = df[TARGET].astype(int)
    df[ID_COL] = df[ID_COL].astype(int)
    return df.reset_index(drop=True)


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds behavioural features. Statistics are learned in ``fit`` only."""

    def fit(self, X: pd.DataFrame, y=None):
        self.support_median_ = float(X["support_calls"].median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[RAW_FEATURES].copy()
        X["spend_per_tenure"] = X["total_spend"] / (X["tenure"] + 1)
        X["engagement_score"] = X["usage_frequency"] / (X["last_interaction"] + 1)
        X["high_support_user"] = (X["support_calls"] > self.support_median_).astype(np.int8)
        X["payment_risk"] = X["payment_delay"] * X["support_calls"]
        return X
