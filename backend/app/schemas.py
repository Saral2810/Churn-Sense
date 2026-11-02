from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

Gender = Literal["Female", "Male"]
Subscription = Literal["Basic", "Standard", "Premium"]
Contract = Literal["Monthly", "Quarterly", "Annual"]
RiskTier = Literal["low", "medium", "high", "critical"]


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 42,
                "gender": "Female",
                "tenure": 8,
                "usage_frequency": 6,
                "support_calls": 7,
                "payment_delay": 24,
                "subscription_type": "Basic",
                "contract_length": "Monthly",
                "total_spend": 320,
                "last_interaction": 21,
            }
        }
    )

    customer_id: Optional[str] = Field(None, max_length=64)
    age: float = Field(..., ge=18, le=100)
    gender: Gender
    tenure: float = Field(..., ge=0, le=120, description="Months as a customer")
    usage_frequency: float = Field(..., ge=0, le=60, description="Sessions in the last month")
    support_calls: float = Field(..., ge=0, le=50)
    payment_delay: float = Field(..., ge=0, le=120, description="Days")
    subscription_type: Subscription
    contract_length: Contract
    total_spend: float = Field(..., ge=0, le=100_000)
    last_interaction: float = Field(..., ge=0, le=365, description="Days since last interaction")


class Driver(BaseModel):
    feature: str
    label: str
    value: Union[float, str]
    contribution: float = Field(description="Log-odds shift vs. a typical customer (+ = towards churn)")
    share: float = Field(description="Share of total absolute contribution, 0-1")


class Recommendation(BaseModel):
    title: str
    detail: str
    priority: Literal["low", "medium", "high"]


class Prediction(BaseModel):
    customer_id: Optional[str] = None
    churn_probability: float
    will_churn: bool
    risk_tier: RiskTier
    threshold: float
    segment: str
    drivers: list[Driver]
    recommendations: list[Recommendation]
    model_version: str


class BatchRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(..., min_length=1)
    explain: bool = Field(False, description="Include full drivers + recommendations per row")


class BatchItem(BaseModel):
    row: int
    customer_id: Optional[str] = None
    churn_probability: float
    will_churn: bool
    risk_tier: RiskTier
    segment: str
    top_driver: Optional[str] = None
    total_spend: float


class RowError(BaseModel):
    row: int
    error: str


class BatchSummary(BaseModel):
    rows: int
    scored: int
    failed: int
    predicted_churners: int
    avg_probability: float
    revenue_at_risk: float
    tiers: dict[str, int]


class BatchResponse(BaseModel):
    summary: BatchSummary
    results: list[BatchItem]
    errors: list[RowError] = []
    model_version: str


class RoiPoint(BaseModel):
    threshold: float
    net_profit: float


class RoiResult(BaseModel):
    threshold: float
    cohort_size: int
    targeted: int
    true_positives: int
    false_positives: int
    missed_churners: int
    customers_saved: float
    revenue_protected: float
    campaign_cost: float
    net_profit: float
    roi: Optional[float]
    optimal_threshold: float
    optimal_profit: float
    target_everyone_profit: float
    curve: list[RoiPoint]
