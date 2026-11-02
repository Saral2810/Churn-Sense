"""Train, evaluate and package the churn model plus dashboard analytics.

Usage (from ``backend/``)::

    python -m ml.train --data ../data/customer_churn.csv --out artifacts

Outputs in ``--out``:
    model.joblib      single sklearn Pipeline + segmenter + metadata
    metrics.json      holdout metrics, curves, importances, model comparison
    analytics.json    business KPIs, churn breakdowns, segments, insights
    customers.json    scored holdout sample for the at-risk table
    scores.npz        holdout labels + probabilities for the ROI simulator
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.features import (
    CATEGORIES,
    FEATURE_LABELS,
    ID_COL,
    MODEL_NUMERIC,
    RAW_CATEGORICAL,
    RAW_FEATURES,
    RAW_NUMERIC,
    TARGET,
    ChurnFeatureEngineer,
    load_raw,
)

log = logging.getLogger("train")
SEED = 42
DEFAULT_RETENTION_COST = 300.0
DEFAULT_REVENUE_SAVED = 2000.0
SEGMENT_FEATURES = ["total_spend", "usage_frequency", "tenure"]
RISK_TIERS = [(0.75, "critical"), (0.5, "high"), (0.25, "medium"), (0.0, "low")]


def risk_tier(p: float) -> str:
    return next(name for cutoff, name in RISK_TIERS if p >= cutoff)


def build_pipeline(model) -> Pipeline:
    preprocess = ColumnTransformer(
        [
            ("num", StandardScaler(), MODEL_NUMERIC),
            (
                "cat",
                OneHotEncoder(
                    categories=[CATEGORIES[c] for c in RAW_CATEGORICAL],
                    handle_unknown="ignore",
                ),
                RAW_CATEGORICAL,
            ),
        ]
    )
    return Pipeline([("features", ChurnFeatureEngineer()), ("preprocess", preprocess), ("model", model)])


# Ordered by preference: on an AUC tie the earlier (smaller, faster) model wins.
CANDIDATES = {
    "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, early_stopping=True, random_state=SEED
    ),
    "logistic_regression": lambda: LogisticRegression(max_iter=1000),
    "random_forest": lambda: RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=SEED),
}


def profit_curve(y: np.ndarray, p: np.ndarray, cost: float, value: float, steps: int = 99):
    thresholds = np.linspace(0.01, 0.99, steps)
    rows = []
    for t in thresholds:
        pred = p >= t
        tp = int((pred & (y == 1)).sum())
        targeted = int(pred.sum())
        rows.append((float(t), tp * value - targeted * cost, tp, targeted))
    return rows


def optimal_threshold(curve) -> float:
    """Most profitable threshold; near-ties (within 0.1%) resolve towards 0.5."""
    top = max(r[1] for r in curve)
    near = [r[0] for r in curve if r[1] >= top - abs(top) * 0.001]
    return min(near, key=lambda t: abs(t - 0.5))


def downsample_curve(x: np.ndarray, y: np.ndarray, n: int = 120) -> list[dict]:
    idx = np.unique(np.linspace(0, len(x) - 1, min(n, len(x))).astype(int))
    return [{"x": round(float(x[i]), 4), "y": round(float(y[i]), 4)} for i in idx]


def band_stats(df: pd.DataFrame, col: str, bins: list, labels: list[str]) -> list[dict]:
    banded = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    return group_stats(df.assign(_band=banded), "_band")


def group_stats(df: pd.DataFrame, col: str) -> list[dict]:
    g = df.groupby(col, observed=True).agg(
        customers=(TARGET, "size"), churn_rate=(TARGET, "mean"), avg_spend=("total_spend", "mean")
    )
    return [
        {
            "name": str(k),
            "customers": int(r.customers),
            "churn_rate": round(float(r.churn_rate), 4),
            "avg_spend": round(float(r.avg_spend), 2),
        }
        for k, r in g.iterrows()
    ]


def fit_segmenter(train: pd.DataFrame):
    scaler = StandardScaler().fit(train[SEGMENT_FEATURES])
    kmeans = KMeans(n_clusters=4, random_state=SEED, n_init=10).fit(scaler.transform(train[SEGMENT_FEATURES]))
    centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=SEGMENT_FEATURES)
    med = train[SEGMENT_FEATURES].median()
    names, used = {}, set()
    # Name clusters from their centroid relative to the median customer.
    for i in centers.sort_values("total_spend", ascending=False).index:
        c = centers.loc[i]
        if c.total_spend <= med.total_spend:
            name = "Budget Drifters"
        elif c.tenure <= med.tenure:
            name = "New High-Spenders"
        elif c.usage_frequency > med.usage_frequency:
            name = "Power Loyalists"
        else:
            name = "Quiet Loyalists"
        base, n = name, 2
        while name in used:
            name, n = f"{base} {n}", n + 1
        used.add(name)
        names[int(i)] = name
    return {"scaler": scaler, "kmeans": kmeans, "names": names, "centers": centers}


def build_analytics(df: pd.DataFrame, seg: dict, segments: np.ndarray) -> dict:
    churned = df[df[TARGET] == 1]
    rate = lambda mask: float(df.loc[mask, TARGET].mean())  # noqa: E731

    seg_df = df.assign(segment=[seg["names"][s] for s in segments])
    segment_rows = []
    for name, g in seg_df.groupby("segment"):
        segment_rows.append(
            {
                "name": name,
                "customers": int(len(g)),
                "share": round(len(g) / len(df), 4),
                "churn_rate": round(float(g[TARGET].mean()), 4),
                "avg_spend": round(float(g.total_spend.mean()), 2),
                "avg_tenure": round(float(g.tenure.mean()), 1),
                "avg_usage": round(float(g.usage_frequency.mean()), 1),
            }
        )
    segment_rows.sort(key=lambda r: -r["avg_spend"])

    spend_bins = np.linspace(df.total_spend.min(), df.total_spend.max(), 21)
    hist_c, _ = np.histogram(churned.total_spend, bins=spend_bins)
    hist_r, _ = np.histogram(df[df[TARGET] == 0].total_spend, bins=spend_bins)

    monthly, annual = rate(df.contract_length == "Monthly"), rate(df.contract_length == "Annual")
    insights = [
        {
            "title": "Contract length is the #1 lever",
            "detail": f"Monthly contracts churn at {monthly:.0%} vs {annual:.0%} for annual plans. "
            "Converting monthly customers to longer terms is the highest-ROI play.",
            "severity": "critical",
        },
        {
            "title": "Support friction predicts exit",
            "detail": f"Customers with more than 5 support calls churn at "
            f"{rate(df.support_calls > 5):.0%} (vs {rate(df.support_calls <= 5):.0%}). "
            "Route them to priority support early.",
            "severity": "high",
        },
        {
            "title": "Late payments are an early warning",
            "detail": f"A payment delay above 20 days pushes churn to {rate(df.payment_delay > 20):.0%}. "
            "Trigger flexible-payment offers at day 10.",
            "severity": "high",
        },
        {
            "title": "Low spenders are fragile",
            "detail": f"Customers spending under 500 churn at {rate(df.total_spend < 500):.0%} "
            f"vs {rate(df.total_spend >= 500):.0%} above. Upsell bundles increase stickiness.",
            "severity": "medium",
        },
    ]

    return {
        "kpis": {
            "customers": int(len(df)),
            "churned": int(len(churned)),
            "churn_rate": round(float(df[TARGET].mean()), 4),
            "avg_tenure": round(float(df.tenure.mean()), 1),
            "avg_spend": round(float(df.total_spend.mean()), 2),
            "revenue_at_risk": round(float(churned.total_spend.sum()), 2),
            "avg_support_calls": round(float(df.support_calls.mean()), 2),
        },
        "by_contract": group_stats(df, "contract_length"),
        "by_subscription": group_stats(df, "subscription_type"),
        "by_gender": group_stats(df, "gender"),
        "by_age": band_stats(df, "age", [0, 25, 35, 45, 55, 120], ["18-25", "26-35", "36-45", "46-55", "56+"]),
        "by_tenure": band_stats(
            df, "tenure", [0, 6, 12, 24, 36, 48, 60], ["0-6", "7-12", "13-24", "25-36", "37-48", "49-60"]
        ),
        "by_support_calls": group_stats(df.assign(support_calls=df.support_calls.astype(int)), "support_calls"),
        "by_payment_delay": band_stats(
            df, "payment_delay", [-1, 5, 10, 15, 20, 25, 30], ["0-5", "6-10", "11-15", "16-20", "21-25", "26-30"]
        ),
        "by_last_interaction": band_stats(
            df, "last_interaction", [0, 5, 10, 15, 20, 25, 30], ["1-5", "6-10", "11-15", "16-20", "21-25", "26-30"]
        ),
        "spend_distribution": [
            {
                "bucket": f"{int(spend_bins[i])}",
                "churned": int(hist_c[i]),
                "retained": int(hist_r[i]),
            }
            for i in range(len(hist_c))
        ],
        "segments": segment_rows,
        "insights": insights,
    }


def evaluate(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 5),
        "pr_auc": round(float(average_precision_score(y, p)), 5),
        "accuracy": round(float(accuracy_score(y, pred)), 5),
        "precision": round(float(precision_score(y, pred)), 5),
        "recall": round(float(recall_score(y, pred)), 5),
        "f1": round(float(f1_score(y, pred)), 5),
        "brier": round(float(brier_score_loss(y, p)), 5),
    }


def main(data_path: str, out_dir: str, candidates: list[str]) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_hash = hashlib.sha256(Path(data_path).read_bytes()).hexdigest()

    df = load_raw(data_path)
    log.info("Loaded %d rows (churn rate %.3f)", len(df), df[TARGET].mean())

    train, test = train_test_split(df, test_size=0.2, stratify=df[TARGET], random_state=SEED)
    X_train, y_train = train[RAW_FEATURES], train[TARGET].to_numpy()
    X_test, y_test = test[RAW_FEATURES], test[TARGET].to_numpy()

    # ---- model comparison -------------------------------------------------
    comparison, fitted = [], {}
    for name in candidates:
        t0 = time.perf_counter()
        pipe = build_pipeline(CANDIDATES[name]()).fit(X_train, y_train)
        train_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        proba = pipe.predict_proba(X_test)[:, 1]
        infer_ms = (time.perf_counter() - t0) / len(X_test) * 1e6  # µs per row
        tmp = out / f".tmp_{name}.joblib"
        joblib.dump(pipe, tmp, compress=3)
        size_kb = tmp.stat().st_size / 1024
        tmp.unlink()
        metrics = evaluate(y_test, proba, 0.5)
        comparison.append(
            {
                "model": name,
                **metrics,
                "train_seconds": round(train_s, 1),
                "inference_us_per_row": round(infer_ms, 2),
                "artifact_kb": round(size_kb, 1),
            }
        )
        fitted[name] = (pipe, proba)
        log.info("%-24s AUC=%.5f  size=%.0fKB  train=%.1fs", name, metrics["roc_auc"], size_kb, train_s)

    best = min(comparison, key=lambda r: (-round(r["roc_auc"], 4), candidates.index(r["model"])))["model"]
    pipe, proba = fitted[best]
    log.info("Selected %s", best)

    # ---- threshold tuned for profit ---------------------------------------
    curve = profit_curve(y_test, proba, DEFAULT_RETENTION_COST, DEFAULT_REVENUE_SAVED)
    best_t = optimal_threshold(curve)

    # ---- explainability ----------------------------------------------------
    sample = test.sample(n=min(20000, len(test)), random_state=SEED)
    perm = permutation_importance(
        pipe, sample[RAW_FEATURES], sample[TARGET], scoring="roc_auc", n_repeats=3, random_state=SEED, n_jobs=-1
    )
    importance = sorted(
        (
            {"feature": f, "label": FEATURE_LABELS[f], "importance": round(float(m), 5)}
            for f, m in zip(RAW_FEATURES, perm.importances_mean)
        ),
        key=lambda r: -r["importance"],
    )

    fpr, tpr, _ = roc_curve(y_test, proba)
    prec, rec, _ = precision_recall_curve(y_test, proba)
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
    cm = confusion_matrix(y_test, (proba >= best_t).astype(int))

    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    version = f"{datetime.now(timezone.utc):%Y.%m.%d}-{best[:3]}-{data_hash[:7]}"

    # ---- segmentation -------------------------------------------------------
    seg = fit_segmenter(train)
    all_segments = seg["kmeans"].predict(seg["scaler"].transform(df[SEGMENT_FEATURES]))

    reference = {c: float(train[c].median()) for c in RAW_NUMERIC}
    reference.update({c: str(train[c].mode().iloc[0]) for c in RAW_CATEGORICAL})

    bundle = {
        "pipeline": pipe,
        "segmenter": {"scaler": seg["scaler"], "kmeans": seg["kmeans"], "names": seg["names"]},
        "segment_features": SEGMENT_FEATURES,
        "reference": reference,
        "threshold": float(best_t),
        "model_name": best,
        "version": version,
        "trained_at": trained_at,
    }
    joblib.dump(bundle, out / "model.joblib", compress=3)

    metrics_doc = {
        "model_name": best,
        "version": version,
        "trained_at": trained_at,
        "threshold": round(float(best_t), 3),
        "holdout": evaluate(y_test, proba, best_t),
        "holdout_at_0_5": evaluate(y_test, proba, 0.5),
        "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
        "roc_curve": downsample_curve(fpr, tpr),
        "pr_curve": downsample_curve(rec[::-1], prec[::-1]),
        "calibration": [
            {"predicted": round(float(m), 4), "observed": round(float(f), 4)} for m, f in zip(mean_pred, frac_pos)
        ],
        "feature_importance": importance,
        "comparison": comparison,
        "data": {
            "rows": int(len(df)),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "positive_rate": round(float(df[TARGET].mean()), 4),
            "sha256": data_hash,
        },
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "notes": [
            "The public training dataset is close to perfectly separable (for example, all Monthly "
            "contracts churn), so near-perfect holdout scores are expected and will not carry over "
            "to noisier real-world data.",
            "The decision threshold is tuned for profit (retention cost 300, revenue saved 2000), not accuracy.",
        ],
    }
    (out / "metrics.json").write_text(json.dumps(metrics_doc, indent=2))
    (out / "analytics.json").write_text(json.dumps(build_analytics(df, seg, all_segments), indent=2))

    np.savez_compressed(out / "scores.npz", y=y_test.astype(np.int8), p=proba.astype(np.float32))

    # ---- scored customer sample (holdout only, so no train leakage) --------
    pick = test.sample(n=min(800, len(test)), random_state=SEED + 1)
    pick_p = pipe.predict_proba(pick[RAW_FEATURES])[:, 1]
    pick_seg = seg["kmeans"].predict(seg["scaler"].transform(pick[SEGMENT_FEATURES]))
    customers = [
        {
            "customer_id": int(r[ID_COL]),
            **{c: (r[c] if c in RAW_CATEGORICAL else float(r[c])) for c in RAW_FEATURES},
            "churn_probability": round(float(p), 4),
            "risk_tier": risk_tier(float(p)),
            "segment": seg["names"][int(s)],
            "actual_churn": int(r[TARGET]),
        }
        for (_, r), p, s in zip(pick.iterrows(), pick_p, pick_seg)
    ]
    customers.sort(key=lambda c: (-c["churn_probability"], -c["total_spend"]))
    (out / "customers.json").write_text(json.dumps(customers))
    log.info("Artifacts written to %s (version %s)", out.resolve(), version)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="../data/customer_churn.csv")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument(
        "--models",
        nargs="+",
        default=list(CANDIDATES),
        choices=list(CANDIDATES),
        help="candidate models to benchmark (first listed wins ties)",
    )
    args = ap.parse_args()
    main(args.data, args.out, args.models)
