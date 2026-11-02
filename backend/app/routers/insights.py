from typing import Literal, Optional

from fastapi import APIRouter, Depends, Query

from app.deps import get_service
from app.model_service import ModelService
from app.schemas import RoiResult

router = APIRouter(tags=["insights"])


@router.get("/analytics", summary="Business KPIs, churn breakdowns, segments and insights")
def analytics(service: ModelService = Depends(get_service)):
    return service.analytics


@router.get("/model", summary="Model card: holdout metrics, curves, feature importance")
def model_card(service: ModelService = Depends(get_service)):
    return service.metrics


@router.get("/roi", response_model=RoiResult, summary="Simulate a retention campaign on the holdout set")
def roi(
    retention_cost: float = Query(300, ge=0, le=100_000, description="Cost of contacting one customer"),
    revenue_saved: float = Query(2000, ge=0, le=1_000_000, description="Value of retaining one churner"),
    success_rate: float = Query(1.0, ge=0, le=1, description="Share of contacted churners actually retained"),
    threshold: float = Query(0.5, ge=0, le=1),
    service: ModelService = Depends(get_service),
):
    return service.roi(retention_cost, revenue_saved, success_rate, threshold)


@router.get("/customers", summary="Scored holdout customers, highest risk first")
def customers(
    tier: Optional[Literal["low", "medium", "high", "critical"]] = None,
    segment: Optional[str] = None,
    q: Optional[str] = Query(None, max_length=32, description="Search by customer id"),
    sort: Literal["risk", "spend", "value_at_risk"] = "risk",
    limit: int = Query(25, ge=1, le=200),
    offset: int = Query(0, ge=0),
    service: ModelService = Depends(get_service),
):
    return service.list_customers(tier, segment, q, sort, limit, offset)
