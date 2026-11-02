import type { RiskTier } from "./types";

const compact = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 });
const whole = new Intl.NumberFormat("en-US");
const money = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
const moneyCompact = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  notation: "compact",
  maximumFractionDigits: 1,
});

export const fmt = {
  num: (n: number) => whole.format(Math.round(n)),
  compact: (n: number) => compact.format(n),
  money: (n: number) => money.format(n),
  moneyCompact: (n: number) => moneyCompact.format(n),
  pct: (n: number, digits = 1) => `${(n * 100).toFixed(digits)}%`,
  dec: (n: number, digits = 3) => n.toFixed(digits),
};

export const TIER_STYLES: Record<RiskTier, { label: string; color: string; chip: string; dot: string }> = {
  low: { label: "Low", color: "#8CC5C0", chip: "bg-low-soft text-low-text ring-1 ring-low/50", dot: "bg-low" },
  medium: { label: "Medium", color: "#EBC57E", chip: "bg-med-soft text-med-text ring-1 ring-med/50", dot: "bg-med" },
  high: { label: "High", color: "#F0B08A", chip: "bg-high-soft text-high-text ring-1 ring-high/50", dot: "bg-high" },
  critical: { label: "Critical", color: "#E59C9C", chip: "bg-crit-soft text-crit-text ring-1 ring-crit/50", dot: "bg-crit" },
};

export const tierOf = (p: number): RiskTier => (p >= 0.75 ? "critical" : p >= 0.5 ? "high" : p >= 0.25 ? "medium" : "low");

/** Map a churn rate (0-1) onto a green → amber → rose ramp for heat-style bars. */
export function rateColor(rate: number): string {
  if (rate >= 0.8) return "#E59C9C";
  if (rate >= 0.6) return "#F0B08A";
  if (rate >= 0.4) return "#EBC57E";
  return "#8CC5C0";
}
