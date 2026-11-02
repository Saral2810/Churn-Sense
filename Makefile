.PHONY: setup train api web test lint

setup:            ## Create venv + install backend and frontend deps
	python3 -m venv .venv && .venv/bin/pip install -r backend/requirements-dev.txt
	cd frontend && npm install

train:            ## Retrain the model and regenerate dashboard artifacts
	cd backend && ../.venv/bin/python -m ml.train --data ../data/customer_churn.csv --out artifacts

api:              ## Run the API on :8000 with reload
	cd backend && ../.venv/bin/uvicorn app.main:app --reload --port 8000

web:              ## Run the dashboard on :5173
	cd frontend && npm run dev

test:             ## Backend tests + frontend type-check
	cd backend && ../.venv/bin/pytest -q
	cd frontend && npm run typecheck

lint:
	cd backend && ../.venv/bin/ruff check . && ../.venv/bin/ruff format --check .
