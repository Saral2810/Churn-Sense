export type RiskTier = "low" | "medium" | "high" | "critical";
export type Gender = "Female" | "Male";
export type Subscription = "Basic" | "Standard" | "Premium";
export type Contract = "Monthly" | "Quarterly" | "Annual";

export interface CustomerInput {
  customer_id?: string;
  age: number;
  gender: Gender;
  tenure: number;
  usage_frequency: number;
  support_calls: number;
  payment_delay: number;
  subscription_type: Subscription;
  contract_length: Contract;
  total_spend: number;
  last_interaction: number;
}

export interface Driver {
  feature: string;
  label: string;
  value: number | string;
  contribution: number;
  share: number;
}

export interface Recommendation {
  title: string;
  detail: string;
  priority: "low" | "medium" | "high";
}

export interface Prediction {
  customer_id: string | null;
  churn_probability: number;
  will_churn: boolean;
  risk_tier: RiskTier;
  threshold: number;
  segment: string;
  drivers: Driver[];
  recommendations: Recommendation[];
  model_version: string;
}

export interface GroupStat {
  name: string;
  customers: number;
  churn_rate: number;
  avg_spend: number;
}

export interface Segment {
  name: string;
  customers: number;
  share: number;
  churn_rate: number;
  avg_spend: number;
  avg_tenure: number;
  avg_usage: number;
}

export interface Insight {
  title: string;
  detail: string;
  severity: "critical" | "high" | "medium";
}

export interface Analytics {
  kpis: {
    customers: number;
    churned: number;
    churn_rate: number;
    avg_tenure: number;
    avg_spend: number;
    revenue_at_risk: number;
    avg_support_calls: number;
  };
  by_contract: GroupStat[];
  by_subscription: GroupStat[];
  by_gender: GroupStat[];
  by_age: GroupStat[];
  by_tenure: GroupStat[];
  by_support_calls: GroupStat[];
  by_payment_delay: GroupStat[];
  by_last_interaction: GroupStat[];
  spend_distribution: { bucket: string; churned: number; retained: number }[];
  segments: Segment[];
  insights: Insight[];
}

export interface Metrics {
  roc_auc: number;
  pr_auc: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  brier: number;
}

export interface ModelCard {
  model_name: string;
  version: string;
  trained_at: string;
  threshold: number;
  holdout: Metrics;
  holdout_at_0_5: Metrics;
  confusion_matrix: { tn: number; fp: number; fn: number; tp: number };
  roc_curve: { x: number; y: number }[];
  pr_curve: { x: number; y: number }[];
  calibration: { predicted: number; observed: number }[];
  feature_importance: { feature: string; label: string; importance: number }[];
  comparison: (Metrics & {
    model: string;
    train_seconds: number;
    inference_us_per_row: number;
    artifact_kb: number;
  })[];
  data: { rows: number; train_rows: number; test_rows: number; positive_rate: number; sha256: string };
  environment: Record<string, string>;
  notes: string[];
}

export interface RoiResult {
  threshold: number;
  cohort_size: number;
  targeted: number;
  true_positives: number;
  false_positives: number;
  missed_churners: number;
  customers_saved: number;
  revenue_protected: number;
  campaign_cost: number;
  net_profit: number;
  roi: number | null;
  optimal_threshold: number;
  optimal_profit: number;
  target_everyone_profit: number;
  curve: { threshold: number; net_profit: number }[];
}

export interface ScoredCustomer extends CustomerInput {
  customer_id: string;
  churn_probability: number;
  risk_tier: RiskTier;
  segment: string;
  actual_churn: number;
}

export interface CustomerPage {
  total: number;
  tiers: Partial<Record<RiskTier, number>>;
  items: ScoredCustomer[];
}

export interface BatchItem {
  row: number;
  customer_id: string | null;
  churn_probability: number;
  will_churn: boolean;
  risk_tier: RiskTier;
  segment: string;
  top_driver: string | null;
  total_spend: number;
}

export interface BatchResponse {
  summary: {
    rows: number;
    scored: number;
    failed: number;
    predicted_churners: number;
    avg_probability: number;
    revenue_at_risk: number;
    tiers: Record<RiskTier, number>;
  };
  results: BatchItem[];
  errors: { row: number; error: string }[];
  model_version: string;
}

export interface Health {
  status: string;
  model_version: string;
  model_name: string;
  trained_at: string;
  environment: string;
}
