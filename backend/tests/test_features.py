import pandas as pd

from ml.features import ENGINEERED, ChurnFeatureEngineer
from tests.conftest import HIGH_RISK, LOW_RISK


def test_engineered_columns_and_values():
    df = pd.DataFrame([HIGH_RISK, LOW_RISK])
    out = ChurnFeatureEngineer().fit(df).transform(df)
    assert set(ENGINEERED) <= set(out.columns)
    assert out.loc[0, "payment_risk"] == 24 * 7
    assert out.loc[1, "spend_per_tenure"] == 900 / 49


def test_median_is_learned_from_fit_data_only():
    fe = ChurnFeatureEngineer().fit(pd.DataFrame([LOW_RISK, LOW_RISK]))
    assert fe.support_median_ == 1
    scored = fe.transform(pd.DataFrame([HIGH_RISK]))
    assert scored.loc[0, "high_support_user"] == 1
