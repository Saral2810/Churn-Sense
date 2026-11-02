"""Loads trained artifacts once and serves predictions, explanations and ROI maths."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from ml.features import FEATURE_LABELS, RAW_CATEGORICAL, RAW_FEATURES

log = logging.getLogger(__name__)

RISK_TIERS = [(0.75, "critical"), (0.5, "high"), (0.25, "medium"), (0.0, "low")]
REQUIRED_ARTIFACTS = ["model.joblib", "metrics.json", "analytics.json", "customers.json", "scores.npz"]


def risk_tier(p: float) -> str:
    return next(name for cutoff, name in RISK_TIERS if p >= cutoff)


class ModelService:
    def __init__(self, artifacts_dir: Path):
        missing = [f for f in REQUIRED_ARTIFACTS if not (artifacts_dir / f).exists()]
        if missing:
            raise RuntimeError(f"Missing artifacts in {artifacts_dir}: {missing}. Run `python -m ml.train` first.")
        bundle = joblib.load(artifacts_dir / "model.joblib")
        self.pipeline = bundle["pipeline"]
        self.segmenter = bundle["segmenter"]
        self.segment_features: list[str] = bundle["segment_features"]
        self.reference: dict[str, Any] = bundle["reference"]
        self.threshold: float = bundle["threshold"]
        self.model_name: str = bundle["model_name"]
        self.version: str = bundle["version"]
        self.trained_at: str = bundle["trained_at"]

        self.metrics = json.loads((artifacts_dir / "metrics.json").read_text())
        self.analytics = json.loads((artifacts_dir / "analytics.json").read_text())
        self.customers: list[dict] = json.loads((artifacts_dir / "customers.json").read_text())
        scores = np.load(artifacts_dir / "scores.npz")
        self._y, self._p = scores["y"].astype(int), scores["p"].astype(float)
        log.info("Loaded model %s (%s)", self.version, self.model_name)

    # ------------------------------------------------------------------ scoring
    def _frame(self, rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows, columns=RAW_FEATURES)

    def _logit(self, X: pd.DataFrame) -> np.ndarray:
        if hasattr(self.pipeline, "decision_function"):
            try:
                return np.asarray(self.pipeline.decision_function(X), dtype=float)
            except AttributeError:  # final estimator has no decision_function
                pass
        p = np.clip(self.pipeline.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def _segments(self, X: pd.DataFrame) -> list[str]:
        ids = self.segmenter["kmeans"].predict(self.segmenter["scaler"].transform(X[self.segment_features]))
        return [self.segmenter["names"][int(i)] for i in ids]

    def _contributions(self, X: pd.DataFrame) -> np.ndarray:
        """Counterfactual log-odds contributions, shape (n_rows, n_features).

        Two-sided, a cheap Shapley-style approximation in log-odds:
          * remove: customer's log-odds minus log-odds with feature i reset to a
            *typical* customer's value (median/mode of training data);
          * add:    typical customer's log-odds with feature i set to this
            customer's value, minus the typical customer's log-odds.
        Averaging both matters because churn rules overlap (e.g. Monthly contract
        OR >5 support calls): removing one sufficient cause alone changes nothing.
        """
        n, k = len(X), len(RAW_FEATURES)
        ref_row = pd.DataFrame([self.reference], columns=RAW_FEATURES)
        ref_logit = self._logit(ref_row)[0]
        removed = pd.concat([X.assign(**{f: self.reference[f]}) for f in RAW_FEATURES], ignore_index=True)
        added = pd.concat(
            [pd.concat([ref_row] * n, ignore_index=True).assign(**{f: X[f].to_numpy()}) for f in RAW_FEATURES],
            ignore_index=True,
        )
        logits = self._logit(pd.concat([X, removed, added], ignore_index=True))
        base, rem, add = logits[:n], logits[n : n + n * k], logits[n + n * k :]
        remove_effect = base[:, None] - rem.reshape(k, n).T
        add_effect = add.reshape(k, n).T - ref_logit
        return (remove_effect + add_effect) / 2

    def score(self, rows: list[dict], explain: bool = True) -> list[dict]:
        X = self._frame(rows)
        proba = self.pipeline.predict_proba(X)[:, 1]
        segments = self._segments(X)
        contrib = self._contributions(X)
        out = []
        for i, row in enumerate(rows):
            p = float(proba[i])
            c = contrib[i]
            total = float(np.abs(c).sum()) or 1.0
            order = np.argsort(-np.abs(c))
            drivers = [
                {
                    "feature": RAW_FEATURES[j],
                    "label": FEATURE_LABELS[RAW_FEATURES[j]],
                    "value": row[RAW_FEATURES[j]],
                    "contribution": round(float(c[j]), 4),
                    "share": round(float(abs(c[j]) / total), 4),
                }
                for j in order
                if abs(c[j]) > 1e-6
            ]
            top = next((d["label"] for d in drivers if d["contribution"] > 0), None)
            item = {
                "customer_id": row.get("customer_id"),
                "churn_probability": round(p, 4),
                "will_churn": p >= self.threshold,
                "risk_tier": risk_tier(p),
                "threshold": self.threshold,
                "segment": segments[i],
                "top_driver": top,
                "total_spend": float(row["total_spend"]),
                "model_version": self.version,
            }
            if explain:
                item["drivers"] = drivers[:6]
                item["recommendations"] = recommend(row, p)
            out.append(item)
        return out

    # ---------------------------------------------------------------------- ROI
    def roi(self, retention_cost: float, revenue_saved: float, success_rate: float, threshold: float) -> dict:
        y, p = self._y, self._p

        def at(t: float) -> dict:
            pred = p >= t
            tp = int((pred & (y == 1)).sum())
            targeted = int(pred.sum())
            saved = tp * success_rate
            revenue = saved * revenue_saved
            cost = targeted * retention_cost
            return {
                "targeted": targeted,
                "tp": tp,
                "fp": targeted - tp,
                "fn": int(((~pred) & (y == 1)).sum()),
                "saved": saved,
                "revenue": revenue,
                "cost": cost,
                "profit": revenue - cost,
            }

        grid = np.round(np.linspace(0.01, 0.99, 99), 2)
        curve = [{"threshold": float(t), "net_profit": round(at(t)["profit"], 2)} for t in grid]
        top = max(pt["net_profit"] for pt in curve)
        near = [pt["threshold"] for pt in curve if pt["net_profit"] >= top - abs(top) * 0.001]
        best_t = min(near, key=lambda t: abs(t - 0.5))
        r = at(threshold)
        everyone = at(0.0)
        return {
            "threshold": threshold,
            "cohort_size": int(len(y)),
            "targeted": r["targeted"],
            "true_positives": r["tp"],
            "false_positives": r["fp"],
            "missed_churners": r["fn"],
            "customers_saved": round(r["saved"], 1),
            "revenue_protected": round(r["revenue"], 2),
            "campaign_cost": round(r["cost"], 2),
            "net_profit": round(r["profit"], 2),
            "roi": round(r["profit"] / r["cost"], 4) if r["cost"] else None,
            "optimal_threshold": best_t,
            "optimal_profit": round(at(best_t)["profit"], 2),
            "target_everyone_profit": round(everyone["profit"], 2),
            "curve": curve,
        }

    # ---------------------------------------------------------------- customers
    def list_customers(
        self,
        tier: Optional[str],
        segment: Optional[str],
        q: Optional[str],
        sort: str,
        limit: int,
        offset: int,
    ) -> dict:
        items = self.customers
        if tier:
            items = [c for c in items if c["risk_tier"] == tier]
        if segment:
            items = [c for c in items if c["segment"] == segment]
        if q:
            items = [c for c in items if q.strip() in str(c["customer_id"])]
        if sort == "spend":
            items = sorted(items, key=lambda c: -c["total_spend"])
        elif sort == "value_at_risk":
            items = sorted(items, key=lambda c: -c["total_spend"] * c["churn_probability"])
        tiers: dict[str, int] = {}
        for c in self.customers:
            tiers[c["risk_tier"]] = tiers.get(c["risk_tier"], 0) + 1
        return {"total": len(items), "tiers": tiers, "items": items[offset : offset + limit]}


def recommend(row: dict, p: float) -> list[dict]:
    """Rule-based retention playbook keyed off the behaviours the data shows matter."""
    recs = []
    if row["contract_length"] == "Monthly":
        recs.append(
            (
                "Lock in a longer contract",
                "Offer 2 free months on an annual upgrade. Monthly plans have by far the highest churn.",
                "high",
            )
        )
    if row["support_calls"] > 5:
        recs.append(
            (
                "Escalate to priority support",
                "Assign a dedicated success manager and close all open tickets within 48h.",
                "high",
            )
        )
    if row["payment_delay"] > 15:
        recs.append(
            (
                "Offer a flexible payment plan",
                "Proactively propose split payments or a short grace period before the account lapses.",
                "high" if row["payment_delay"] > 20 else "medium",
            )
        )
    if row["total_spend"] < 500:
        recs.append(
            (
                "Upsell a value bundle",
                "Low-spend customers churn heavily. A discounted bundle increases switching costs.",
                "medium",
            )
        )
    if row["last_interaction"] > 15:
        recs.append(
            (
                "Proactive check-in",
                "No contact for over two weeks. Trigger a personal outreach or a product-tips email.",
                "medium",
            )
        )
    if row["usage_frequency"] < 10:
        recs.append(
            ("Re-engagement campaign", "Low product usage. Send a guided feature tour and usage nudges.", "medium")
        )
    if row["subscription_type"] == "Basic" and p >= 0.5:
        recs.append(("Trial a premium tier", "Offer a 30-day Premium trial to surface more value.", "low"))
    if not recs:
        recs.append(
            ("Nurture & reward loyalty", "Low risk. Enrol in loyalty rewards and ask for a referral or review.", "low")
        )
    return [{"title": t, "detail": d, "priority": pr} for t, d, pr in recs[:4]]


def normalise_record(raw: dict) -> dict:
    """Accept either snake_case keys or the original CSV headers."""
    from ml.features import RAW_TO_API

    out = {}
    for k, v in raw.items():
        key = str(k).strip()
        key = RAW_TO_API.get(key, key.lower().replace(" ", "_"))
        if key == "customerid":
            key = "customer_id"
        out[key] = v
    for c in RAW_CATEGORICAL:
        if isinstance(out.get(c), str):
            out[c] = out[c].strip().title()
    return out
