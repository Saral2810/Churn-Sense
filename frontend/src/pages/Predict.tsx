import clsx from "clsx";
import { CheckCircle2, Lightbulb, Loader2, ShieldAlert, Sparkles } from "lucide-react";
import { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import { Gauge } from "../components/Gauge";
import { Card, PageHeader, RiskBadge, Segmented } from "../components/ui";
import { useDebounced } from "../hooks/useApi";
import { api } from "../lib/api";
import { fmt } from "../lib/format";
import type { Contract, CustomerInput, Gender, Prediction, Subscription } from "../lib/types";

const PRESETS: { name: string; emoji: string; data: CustomerInput }[] = [
  {
    name: "Loyal power user",
    emoji: "💎",
    data: { age: 34, gender: "Male", tenure: 48, usage_frequency: 25, support_calls: 1, payment_delay: 3, subscription_type: "Premium", contract_length: "Annual", total_spend: 920, last_interaction: 3 },
  },
  {
    name: "Wavering mid-tier",
    emoji: "⚖️",
    data: { age: 45, gender: "Female", tenure: 20, usage_frequency: 12, support_calls: 4, payment_delay: 14, subscription_type: "Standard", contract_length: "Quarterly", total_spend: 610, last_interaction: 14 },
  },
  {
    name: "Flight risk",
    emoji: "🚨",
    data: { age: 52, gender: "Female", tenure: 6, usage_frequency: 4, support_calls: 8, payment_delay: 26, subscription_type: "Basic", contract_length: "Monthly", total_spend: 280, last_interaction: 24 },
  },
];

const SLIDERS: { key: keyof CustomerInput; label: string; min: number; max: number; step?: number; unit?: string }[] = [
  { key: "age", label: "Age", min: 18, max: 65 },
  { key: "tenure", label: "Tenure", min: 1, max: 60, unit: "months" },
  { key: "usage_frequency", label: "Usage frequency", min: 1, max: 30, unit: "/ month" },
  { key: "support_calls", label: "Support calls", min: 0, max: 10 },
  { key: "payment_delay", label: "Payment delay", min: 0, max: 30, unit: "days" },
  { key: "total_spend", label: "Total spend", min: 100, max: 1000, step: 10, unit: "$" },
  { key: "last_interaction", label: "Last interaction", min: 1, max: 30, unit: "days ago" },
];

const PRIORITY = {
  high: "bg-crit-soft text-crit-text ring-crit/50",
  medium: "bg-med-soft text-med-text ring-med/50",
  low: "bg-low-soft text-low-text ring-low/50",
};

export default function Predict() {
  const location = useLocation();
  const [form, setForm] = useState<CustomerInput>(
    () => (location.state as { customer?: CustomerInput } | null)?.customer ?? PRESETS[1].data,
  );
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const debounced = useDebounced(form, 200);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    api
      .predict(debounced)
      .then((r) => alive && (setResult(r), setError(null)))
      .catch((e: Error) => alive && setError(e.message))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [debounced]);

  const set = <K extends keyof CustomerInput>(k: K, v: CustomerInput[K]) => setForm((f) => ({ ...f, [k]: v }));
  const maxAbs = Math.max(...(result?.drivers.map((d) => Math.abs(d.contribution)) ?? [1]), 1e-6);

  return (
    <>
      <PageHeader
        eyebrow="Live predictor"
        title={
          <>
            Score any customer <span className="text-accent-text">in real time</span>
          </>
        }
        subtitle="Drag the sliders. The model re-scores instantly, explains which behaviours drive the risk, and suggests a retention play."
      />

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.05fr)_minmax(0,1fr)]">
        <Card title="Customer profile" subtitle="Start from a preset or build your own">
          <div className="mb-6 flex flex-wrap gap-2">
            {PRESETS.map((p) => (
              <button key={p.name} className="btn-ghost px-3 py-1.5 text-xs" onClick={() => setForm(p.data)}>
                <span aria-hidden>{p.emoji}</span> {p.name}
              </button>
            ))}
          </div>

          <div className="grid gap-5 sm:grid-cols-2">
            {SLIDERS.map(({ key, label, min, max, step, unit }) => (
              <label key={key} className="block">
                <div className="mb-2 flex items-baseline justify-between">
                  <span className="field-label">{label}</span>
                  <span className="font-display text-sm font-semibold text-ink tabular-nums">
                    {unit === "$" ? fmt.money(form[key] as number) : form[key]}
                    {unit && unit !== "$" && <span className="ml-1 text-xs font-normal text-ink-faint">{unit}</span>}
                  </span>
                </div>
                <input
                  type="range"
                  min={min}
                  max={max}
                  step={step ?? 1}
                  value={form[key] as number}
                  onChange={(e) => set(key, Number(e.target.value) as never)}
                  aria-label={label}
                />
              </label>
            ))}
          </div>

          <div className="mt-6 flex flex-wrap gap-x-6 gap-y-4">
            <div>
              <div className="field-label mb-2">Contract</div>
              <Segmented<Contract>
                value={form.contract_length}
                onChange={(v) => set("contract_length", v)}
                options={(["Monthly", "Quarterly", "Annual"] as const).map((v) => ({ value: v, label: v }))}
              />
            </div>
            <div>
              <div className="field-label mb-2">Plan</div>
              <Segmented<Subscription>
                value={form.subscription_type}
                onChange={(v) => set("subscription_type", v)}
                options={(["Basic", "Standard", "Premium"] as const).map((v) => ({ value: v, label: v }))}
              />
            </div>
            <div>
              <div className="field-label mb-2">Gender</div>
              <Segmented<Gender>
                value={form.gender}
                onChange={(v) => set("gender", v)}
                options={(["Female", "Male"] as const).map((v) => ({ value: v, label: v }))}
              />
            </div>
          </div>
        </Card>

        <div className="space-y-4">
          <Card>
            {error ? (
              <p className="py-10 text-center text-sm text-crit-text">{error}</p>
            ) : !result ? (
              <div className="grid h-64 place-items-center">
                <Loader2 className="h-6 w-6 animate-spin text-accent-text" />
              </div>
            ) : (
              <div className={clsx("transition-opacity", loading && "opacity-70")}>
                <div className="flex flex-col items-center">
                  <Gauge value={result.churn_probability} threshold={result.threshold} />
                  <div className="mt-2 flex flex-wrap items-center justify-center gap-2">
                    <RiskBadge tier={result.risk_tier} />
                    <span className="chip bg-mist text-ink-soft ring-1 ring-line">{result.segment}</span>
                  </div>
                  <div
                    className={clsx(
                      "mt-4 flex items-center gap-2 rounded-xl px-4 py-2 text-sm font-semibold",
                      result.will_churn ? "bg-crit-soft text-crit-text" : "bg-low-soft text-low-text",
                    )}
                  >
                    {result.will_churn ? <ShieldAlert className="h-4 w-4" /> : <CheckCircle2 className="h-4 w-4" />}
                    {result.will_churn ? "Likely to churn: act now" : "Likely to stay"}
                    <span className="font-normal text-ink-soft">· threshold {fmt.pct(result.threshold, 0)}</span>
                  </div>
                </div>
              </div>
            )}
          </Card>

          {result && (
            <Card title="Why this score?" subtitle="How much each attribute moves the risk compared with a typical customer">
              <ul className="space-y-3">
                {result.drivers.map((d) => {
                  const up = d.contribution > 0;
                  const w = `${(Math.abs(d.contribution) / maxAbs) * 50}%`;
                  return (
                    <li key={d.feature} className="grid grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)] items-center gap-3 text-sm">
                      <div className="min-w-0">
                        <div className="truncate text-ink">{d.label}</div>
                        <div className="text-xs text-ink-faint">{typeof d.value === "number" ? fmt.num(d.value) : d.value}</div>
                      </div>
                      <div className="relative h-6">
                        <div className="absolute inset-y-0 left-1/2 w-px bg-line" />
                        <div
                          className={clsx("absolute top-1 h-4 rounded-md transition-all duration-500", up ? "left-1/2 bg-crit" : "right-1/2 bg-low")}
                          style={{ width: w }}
                        />
                      </div>
                    </li>
                  );
                })}
              </ul>
              <div className="mt-3 flex justify-between text-[11px] text-ink-faint">
                <span>← lowers risk</span>
                <span>raises risk →</span>
              </div>
            </Card>
          )}

          {result && (
            <Card title="Retention playbook" action={<Sparkles className="h-4 w-4 text-accent-text" />}>
              <ul className="space-y-3">
                {result.recommendations.map((r) => (
                  <li key={r.title} className="flex gap-3 rounded-xl border border-line bg-page p-3">
                    <Lightbulb className="mt-0.5 h-4 w-4 shrink-0 text-accent-text" />
                    <div className="min-w-0">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="text-sm font-semibold text-ink">{r.title}</span>
                        <span className={clsx("chip ring-1 py-0.5 text-[10px] uppercase", PRIORITY[r.priority])}>{r.priority}</span>
                      </div>
                      <p className="mt-1 text-xs leading-relaxed text-ink-soft">{r.detail}</p>
                    </div>
                  </li>
                ))}
              </ul>
            </Card>
          )}
        </div>
      </div>
    </>
  );
}
