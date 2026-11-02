# ChurnSense: Customer Churn Intelligence

> Predict which customers are about to leave, understand **why**, and simulate the **ROI** of a
> retention campaign. A FastAPI scoring service plus a React analytics dashboard, ready to deploy on
> **Render** (API) and **Vercel** (dashboard).

| Overview | Live predictor | ROI simulator |
|---|---|---|
| KPIs, churn drivers, segments, insights | Sliders → instant score, drivers, retention playbook | Tune cost, value and threshold, then see the profit curve |

Other pages include **At-Risk Customers** (filterable watchlist → one-click analysis), **Batch Scoring** (CSV upload → scored
download), and a **Model Card** (ROC/PR curves, calibration, confusion matrix, model comparison).

---

## Architecture

```
┌──────────────────────────┐   HTTPS/JSON   ┌───────────────────────────────────────┐
│  frontend/ (Vercel)      │ ─────────────▶ │  backend/ (Render, FastAPI)           │
│  React + Vite + TS       │                │  /api/v1/predict | batch | csv        │
│  Tailwind v4 + Recharts  │ ◀───────────── │  /api/v1/roi | analytics | customers  │
└──────────────────────────┘                │  /api/v1/model   /health   /docs      │
                                            │        │ loads once at startup        │
                                            │        ▼                              │
                                            │  artifacts/ model.joblib (≈0.4 MB)    │
                                            │  metrics / analytics / customers JSON │
                                            └───────────────▲───────────────────────┘
                                                            │ python -m ml.train
                                            data/customer_churn.csv (440k rows)
```

```
.
├── backend/
│   ├── app/            FastAPI app (config, schemas, routers, model service)
│   ├── ml/             features.py (leakage-safe transformer) · train.py (benchmark, tune, package)
│   ├── artifacts/      versioned model + precomputed dashboard data (committed, small)
│   ├── tests/          pytest suite
│   └── Dockerfile
├── frontend/           Vite + React dashboard (vercel.json included)
├── data/               training CSV
├── notebooks/          original exploration notebook
├── render.yaml         Render blueprint
├── docker-compose.yml  full stack locally
└── IMPROVEMENTS.md     audit of the original project and everything changed
```

## Quick start (local)

Requires Python ≥ 3.9 (3.11 recommended) and Node ≥ 20.

```bash
make setup     # venv + pip + npm install
make train     # optional: artifacts are already committed (~25 s)
make api       # http://localhost:8000/docs
make web       # http://localhost:5173  (in a second terminal)
```

Or with Docker: `docker compose up --build`.

Without `make`:

```bash
python3 -m venv .venv && .venv/bin/pip install -r backend/requirements-dev.txt
cd backend && ../.venv/bin/uvicorn app.main:app --reload --port 8000
# new terminal
cd frontend && cp .env.example .env && npm install && npm run dev
```

## Deploy

### 1. API → Render
1. Push this repo to GitHub.
2. Render → **New → Blueprint** → select the repo. `render.yaml` creates `churnsense-api` (free plan).
3. After the first deploy, open `https://<your-api>.onrender.com/health`. It should return `{"status":"ok", ...}`.

*(Alternatively, deploy `backend/Dockerfile` to Render, Railway, Fly.io or Cloud Run.)*

### 2. Dashboard → Vercel
1. Vercel → **Add New Project** → import the repo → set **Root Directory** to `frontend`.
2. Add the environment variable `VITE_API_URL=https://<your-api>.onrender.com`.
3. Deploy. `vercel.json` handles SPA routing, caching and security headers.

### 3. Connect them
In Render, set `CORS_ORIGINS=https://<your-app>.vercel.app` (comma-separate multiple origins).
To allow Vercel preview URLs too, set `CORS_ORIGIN_REGEX=https://<project>-.*\.vercel\.app`.

> Free Render instances sleep after ~15 minutes idle. The dashboard shows a "waking up" screen and retries
> automatically, so the first load after a nap takes about 30–60 seconds.

> **Why not run the API on Vercel too?** scikit-learn + pandas + numpy go beyond what Vercel's Python
> functions comfortably support, and each cold start would re-load the model. A long-lived container on Render
> is simpler and faster.

## API

Interactive docs live at `/docs`. Example:

```bash
curl -X POST http://localhost:8000/api/v1/predict -H 'Content-Type: application/json' -d '{
  "age": 42, "gender": "Female", "tenure": 8, "usage_frequency": 6, "support_calls": 7,
  "payment_delay": 24, "subscription_type": "Basic", "contract_length": "Monthly",
  "total_spend": 320, "last_interaction": 21 }'
```

```jsonc
{
  "churn_probability": 1.0, "will_churn": true, "risk_tier": "critical", "segment": "Budget Drifters",
  "drivers": [{ "label": "Contract length", "value": "Monthly", "contribution": 9.1, "share": 0.31 }, ...],
  "recommendations": [{ "title": "Lock in a longer contract", "priority": "high", ... }, ...],
  "model_version": "2026.11.01-his-a41be7b"
}
```

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/predict` | One customer → probability, tier, segment, drivers, recommendations |
| `POST /api/v1/predict/batch` | JSON list (≤ `MAX_BATCH_ROWS`) |
| `POST /api/v1/predict/csv` | CSV upload. Accepts original dataset headers; per-row validation errors are reported |
| `GET /api/v1/roi` | Campaign profit at a threshold + full profit curve + optimal threshold |
| `GET /api/v1/analytics` | KPIs, churn breakdowns, segments, insights |
| `GET /api/v1/customers` | Scored holdout watchlist (filter by tier/segment/id, sort, paginate) |
| `GET /api/v1/model` | Model card: metrics, curves, importances, benchmark |
| `GET /health` | Liveness + model version |

### Configuration (env vars)

| Variable | Default | |
|---|---|---|
| `CORS_ORIGINS` | `http://localhost:5173` | Comma-separated allowed origins |
| `CORS_ORIGIN_REGEX` | – | Optional regex (Vercel previews) |
| `API_KEY` | – | If set, `/predict*` require an `X-API-Key` header |
| `MAX_BATCH_ROWS` / `MAX_UPLOAD_MB` | `10000` / `5` | Upload limits |
| `LOG_LEVEL` | `INFO` | |

Frontend: `VITE_API_URL` (required in production) and an optional `VITE_API_KEY`. Anything prefixed `VITE_` ends up
in the browser bundle, so treat that key as a speed bump, not a secret.

## ML pipeline

`python -m ml.train` performs these steps:
1. Loads and cleans the CSV and fingerprints it (SHA-256).
2. Makes a stratified 80/20 split.
3. Runs **one sklearn `Pipeline`**: `ChurnFeatureEngineer` (spend/tenure, engagement, support flag, payment risk) → scaler + one-hot → model. Because the whole pipeline is serialized, training and serving cannot drift apart.
4. Benchmarks HistGradientBoosting, Logistic Regression and Random Forest. The best AUC wins, and ties go to the smaller, faster model.
5. Tunes the decision threshold for **profit** rather than accuracy.
6. Computes permutation importance, ROC/PR curves and calibration.
7. Runs KMeans segmentation and gives the segments business names.
8. Writes versioned artifacts plus the precomputed dashboard data.

Per-prediction **drivers** are counterfactual: each feature is swapped for a typical customer's value, and the
change in log-odds is measured. This stays meaningful even when probabilities saturate.

> ⚠️ **About the ~1.0 AUC.** This public dataset is almost perfectly separable (for example, every Monthly contract
> churns). Near-perfect scores here say more about the data than about the model. On real CRM data, expect roughly
> 0.75–0.90 and re-check calibration. See [IMPROVEMENTS.md](IMPROVEMENTS.md).

## Quality

```bash
make test   # pytest (API contract, features, CSV edge cases, auth) + TS type-check
make lint   # ruff
```

GitHub Actions (`.github/workflows/ci.yml`) runs lint, tests and the frontend build on every push and PR.
