import csv
import io

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.deps import get_service, require_api_key
from app.model_service import ModelService, normalise_record
from app.schemas import BatchRequest, BatchResponse, CustomerFeatures, Prediction

router = APIRouter(prefix="/predict", tags=["predictions"], dependencies=[Depends(require_api_key)])


def _batch_response(service: ModelService, rows: list[dict], errors: list[dict], total: int) -> dict:
    scored = service.score(rows, explain=False) if rows else []
    results = [{"row": r.pop("_row"), **s} for r, s in zip(rows, scored)]
    tiers = {t: 0 for t in ("low", "medium", "high", "critical")}
    for s in scored:
        tiers[s["risk_tier"]] += 1
    churners = [s for s in scored if s["will_churn"]]
    return {
        "summary": {
            "rows": total,
            "scored": len(scored),
            "failed": len(errors),
            "predicted_churners": len(churners),
            "avg_probability": round(sum(s["churn_probability"] for s in scored) / len(scored), 4) if scored else 0.0,
            "revenue_at_risk": round(sum(s["total_spend"] for s in churners), 2),
            "tiers": tiers,
        },
        "results": results,
        "errors": errors,
        "model_version": service.version,
    }


@router.post("", response_model=Prediction, summary="Score a single customer with explanations")
def predict(customer: CustomerFeatures, service: ModelService = Depends(get_service)):
    return service.score([customer.model_dump()], explain=True)[0]


@router.post("/batch", response_model=BatchResponse, summary="Score up to MAX_BATCH_ROWS customers")
def predict_batch(
    body: BatchRequest,
    service: ModelService = Depends(get_service),
    settings: Settings = Depends(get_settings),
):
    if len(body.customers) > settings.max_batch_rows:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"Max {settings.max_batch_rows} rows")
    rows = [{**c.model_dump(), "_row": i} for i, c in enumerate(body.customers)]
    return _batch_response(service, rows, [], len(rows))


@router.post("/csv", response_model=BatchResponse, summary="Upload a CSV and score every row")
async def predict_csv(
    file: UploadFile = File(..., description="CSV with the same columns as the training data"),
    service: ModelService = Depends(get_service),
    settings: Settings = Depends(get_settings),
):
    raw = await file.read(int(settings.max_upload_mb * 1024 * 1024) + 1)
    if len(raw) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"File exceeds {settings.max_upload_mb} MB")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "File must be UTF-8 encoded CSV") from None

    reader = csv.DictReader(io.StringIO(text))
    rows, errors, total = [], [], 0
    for i, rec in enumerate(reader):
        total += 1
        if total > settings.max_batch_rows:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"Max {settings.max_batch_rows} rows")
        rec = normalise_record(rec)
        if rec.get("customer_id") not in (None, ""):
            rec["customer_id"] = str(rec["customer_id"]).removesuffix(".0")
        else:
            rec.pop("customer_id", None)
        try:
            rows.append({**CustomerFeatures.model_validate(rec).model_dump(), "_row": i})
        except ValidationError as e:
            first = e.errors()[0]
            errors.append({"row": i, "error": f"{'.'.join(map(str, first['loc']))}: {first['msg']}"})
    if total == 0:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "CSV has no data rows")
    return _batch_response(service, rows, errors, total)
