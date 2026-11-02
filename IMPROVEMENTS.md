# ChurnSense — Audit & Improvement Plan

This document lists everything that kept the original project (a single Jupyter notebook +
two pickles) from being production-grade, and what was changed to fix it.
✅ = implemented in this repo · 🔜 = recommended next step.

## 1. ML correctness & methodology

| # | Issue found | Fix |
|---|-------------|-----|
| 1 | **Data leakage** – `High_Support_User` used the median of the *whole* dataset before the train/test split. | ✅ Feature engineering is now a fitted scikit-learn transformer (`ml/features.py`); the median is learned on the training split only. |
| 2 | **Train/serve skew** – engineered features were computed by hand in the notebook, but the saved `preprocessor.pkl` didn't include them, so any API would have had to re-implement them (and get them wrong). | ✅ One single `Pipeline` (features → preprocessing → model) is serialized, so serving is identical to training. |
| 3 | **SMOTE on a 57/43 dataset** – oversampling wasn't needed, and it was applied before stacking, which cross-validates on synthetic rows. | ✅ Removed. Class balance is handled by threshold tuning instead. |
| 4 | **~1.0 ROC-AUC accepted without question** – this public dataset is almost perfectly separable (e.g. every Monthly contract or >5 support calls churns). Treating 99.99% as "real-world" performance is misleading. | ✅ Documented in the model card and README. Metrics are now reported on a clean holdout set, alongside PR-AUC, Brier score and a confusion matrix. |
| 5 | **Threshold fixed at 0.5** even though the ROI section admits business cost matters. | ✅ The decision threshold is tuned for profit and saved with the model. A live ROI simulator endpoint lets you re-tune it. |
| 6 | **91 MB stacked model** (RF ×200 + XGB + LR) → slow cold starts, larger than free hosting tiers allow, no measurable accuracy gain. | ✅ Replaced by `HistGradientBoostingClassifier` (~0.5 MB, same AUC, millisecond inference). Logistic regression and random forest are still benchmarked in `train.py`. |
| 7 | SHAP plot computed on SMOTE-resampled training data (it explains synthetic rows). | ✅ Permutation importance on real holdout data, plus **per-prediction drivers** (counterfactual contributions) returned by the API. |
| 8 | KMeans segmentation fit on the full dataset with arbitrary cluster ids, no business labels. | ✅ Segments are fit in the training script, given human-readable names, and profiled (size, churn rate, spend). |
| 9 | Notebook "Business insights" mention "monthly charges" and "₹X" — not in the data. | ✅ Insights are now computed from the data and shown in the dashboard. |
| 10 | No reproducibility (no pinned versions, no seeds in some steps, no data fingerprint). | ✅ Pinned `requirements.txt`, fixed seeds, SHA-256 data fingerprint + library versions stored in `metrics.json`. |

## 2. Engineering & code quality

| # | Issue | Fix |
|---|-------|-----|
| 11 | Everything lived in one notebook with `!pip install` cells and Windows paths. | ✅ Modular package: `backend/ml` (training) and `backend/app` (API). The notebook is kept in `notebooks/` for exploration only. |
| 12 | Pickles were committed and loaded with no version check. | ✅ Artifacts carry a model version and training timestamp. The API validates that they exist at startup and fails fast if they don't. Legacy pickles are moved to `legacy/` and git-ignored. |
| 13 | No input validation. | ✅ Pydantic v2 schemas with ranges and enums. Invalid requests get clear 422 responses. |
| 14 | No tests. | ✅ `pytest` suite for the features, the API contract, and batch/CSV scoring. |
| 15 | No linting/formatting. | ✅ `ruff` lint + format for Python; TypeScript strict mode (`noUnused*`, strict null checks) for the frontend. |
| 16 | No CI. | ✅ GitHub Actions: lint + test backend, type-check + build frontend. |

## 3. Productization (API + dashboard)

| # | Added |
|---|-------|
| 17 | ✅ **FastAPI backend** – `/predict`, `/predict/batch`, `/predict/csv`, `/roi`, `/analytics/overview`, `/customers`, `/model`, `/health`, with auto-generated OpenAPI docs at `/docs`. |
| 18 | ✅ **React dashboard** (Vite + TypeScript + Tailwind + Recharts) with these pages: Overview, Live Predictor, At-Risk Customers, ROI Simulator, Batch Scoring (CSV upload → download), Model Card. |
| 19 | ✅ Actionable output – each prediction returns a risk tier, its top drivers, and retention playbook recommendations. |
| 20 | ✅ Cold-start-aware UI (free Render instances sleep): a "waking up the API" state with automatic retries. |

## 4. Security & operations

| # | Added |
|---|-------|
| 21 | ✅ Config through environment variables (`pydantic-settings`), `.env.example` files, no secrets in code. |
| 22 | ✅ CORS allow-list, optional `X-API-Key` auth on scoring endpoints, upload size/row limits. |
| 23 | ✅ Request-ID + latency middleware, structured logging, GZip responses. |
| 24 | ✅ Dockerfile (non-root user, slim image, healthcheck), `render.yaml` blueprint, `vercel.json` with SPA rewrites + security headers. |

## 5. Recommended next steps (not implemented)

- 🔜 **Real-world data**: retrain on data with genuine noise and a time-based split. Expect AUC of roughly 0.75–0.9 and re-check calibration.
- 🔜 **Experiment tracking & registry** (MLflow / Weights & Biases) and scheduled retraining.
- 🔜 **Drift monitoring** (Evidently): compare live input distributions with the training data.
- 🔜 **Persistence**: store predictions in Postgres (Supabase/Neon) for feedback loops and auditing.
- 🔜 **Auth for the dashboard** (Clerk/Auth0) and per-user rate limiting (Redis/Upstash).
- 🔜 **Observability**: Sentry for errors, OpenTelemetry traces.
- 🔜 **Fairness review**: `gender` and `age` are model inputs. Check whether using them is acceptable for your use case, or remove them.
