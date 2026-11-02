import clsx from "clsx";
import { Coins, Crosshair, Target, TrendingUp, UserX, Wand2 } from "lucide-react";
import { useState } from "react";
import { Area, AreaChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Card, ChartTooltip, ErrorState, KpiCard, PageHeader, Skeleton } from "../components/ui";
import { useApi, useDebounced } from "../hooks/useApi";
import { api } from "../lib/api";
import { fmt } from "../lib/format";

interface Params {
  retention_cost: number;
  revenue_saved: number;
  success_rate: number;
  threshold: number;
}

const CONTROLS: { key: keyof Params; label: string; min: number; max: number; step: number; show: (v: number) => string; help: string }[] = [
  { key: "retention_cost", label: "Cost per retention offer", min: 0, max: 1000, step: 10, show: fmt.money, help: "Discount, gift or agent time spent on each targeted customer" },
  { key: "revenue_saved", label: "Value of a retained customer", min: 100, max: 5000, step: 50, show: fmt.money, help: "Expected future revenue kept when a churner stays" },
  { key: "success_rate", label: "Campaign success rate", min: 0.05, max: 1, step: 0.05, show: (v) => fmt.pct(v, 0), help: "Share of contacted churners who actually stay" },
  { key: "threshold", label: "Targeting threshold", min: 0.05, max: 0.95, step: 0.01, show: (v) => fmt.pct(v, 0), help: "Contact customers whose churn probability is at or above this" },
];

export default function Roi() {
  const [params, setParams] = useState<Params>({ retention_cost: 300, revenue_saved: 2000, success_rate: 0.4, threshold: 0.5 });
  const p = useDebounced(params, 200);
  const { data: r, error, reload, loading } = useApi(() => api.roi(p), [p]);

  if (error) return <ErrorState message={error} onRetry={reload} />;

  const best = r && Math.abs(r.optimal_threshold - params.threshold) < 0.005;
  const lift = r ? r.net_profit - r.target_everyone_profit : 0;

  return (
    <>
      <PageHeader
        eyebrow="Campaign economics"
        title={
          <>
            ROI <span className="text-accent-text">simulator</span>
          </>
        }
        subtitle={`Turn predictions into a business decision. Simulated on ${r ? fmt.num(r.cohort_size) : "…"} holdout customers the model has never seen.`}
      />

      <div className="grid gap-4 xl:grid-cols-[360px_minmax(0,1fr)]">
        <Card title="Assumptions">
          <div className="space-y-6">
            {CONTROLS.map((c) => (
              <label key={c.key} className="block">
                <div className="mb-2 flex items-baseline justify-between gap-2">
                  <span className="text-sm font-medium text-ink">{c.label}</span>
                  <span className="font-display text-sm font-semibold text-ink tabular-nums">{c.show(params[c.key])}</span>
                </div>
                <input
                  type="range"
                  min={c.min}
                  max={c.max}
                  step={c.step}
                  value={params[c.key]}
                  onChange={(e) => setParams((s) => ({ ...s, [c.key]: Number(e.target.value) }))}
                  aria-label={c.label}
                />
                <p className="mt-1.5 text-xs text-ink-faint">{c.help}</p>
              </label>
            ))}
            <button
              className="btn-primary w-full"
              disabled={!r || !!best}
              onClick={() => r && setParams((s) => ({ ...s, threshold: r.optimal_threshold }))}
            >
              <Wand2 className="h-4 w-4" />
              {best ? "Using the optimal threshold" : `Use optimal threshold (${r ? fmt.pct(r.optimal_threshold, 0) : "…"})`}
            </button>
          </div>
        </Card>

        <div className={clsx("space-y-4 transition-opacity", loading && r && "opacity-80")}>
          <div className="grid gap-4 sm:grid-cols-2 2xl:grid-cols-4">
            {r ? (
              <>
                <KpiCard
                  label="Net profit"
                  value={<span className={r.net_profit < 0 ? "text-crit-text" : ""}>{fmt.moneyCompact(r.net_profit)}</span>}
                  hint={r.roi !== null ? `${fmt.pct(r.roi, 0)} return on spend` : "No customers targeted"}
                  icon={<Coins className="h-4.5 w-4.5" />}
                  accent="emerald"
                />
                <KpiCard label="Customers targeted" value={fmt.compact(r.targeted)} hint={`${fmt.moneyCompact(r.campaign_cost)} campaign cost`} icon={<Target className="h-4.5 w-4.5" />} />
                <KpiCard label="Churners saved" value={fmt.compact(r.customers_saved)} hint={`${fmt.moneyCompact(r.revenue_protected)} revenue protected`} icon={<TrendingUp className="h-4.5 w-4.5" />} accent="cyan" />
                <KpiCard label="Missed churners" value={fmt.compact(r.missed_churners)} hint={`${fmt.num(r.false_positives)} false alarms`} icon={<UserX className="h-4.5 w-4.5" />} accent="rose" />
              </>
            ) : (
              Array.from({ length: 4 }, (_, i) => <Skeleton key={i} className="h-32" />)
            )}
          </div>

          <Card title="Profit by targeting threshold" subtitle="Where should you draw the line?" action={<Crosshair className="h-4 w-4 text-accent-text" />}>
            {r ? (
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={r.curve} margin={{ top: 10, right: 12, left: 4, bottom: 0 }}>
                  <defs>
                    <linearGradient id="g-profit" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0" stopColor="#6E8FC4" stopOpacity={0.55} />
                      <stop offset="1" stopColor="#6E8FC4" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid vertical={false} />
                  <XAxis dataKey="threshold" type="number" domain={[0, 1]} tickFormatter={(v) => fmt.pct(v, 0)} axisLine={false} tickLine={false} />
                  <YAxis tickFormatter={(v) => fmt.moneyCompact(v)} axisLine={false} tickLine={false} width={64} />
                  <Tooltip
                    content={({ active, payload, label }) => (
                      <ChartTooltip
                        active={active}
                        label={`Threshold ${fmt.pct(Number(label), 0)}`}
                        payload={payload?.map((x) => ({ name: "Net profit", value: Number(x.value), color: "#6E8FC4" }))}
                        formatter={fmt.money}
                      />
                    )}
                  />
                  <ReferenceLine y={0} stroke="#C5CFDF" />
                  <ReferenceLine x={r.optimal_threshold} stroke="#8CC5C0" strokeDasharray="4 4" label={{ value: "optimal", fill: "#8CC5C0", fontSize: 11, position: "insideTopLeft" }} />
                  <ReferenceLine x={params.threshold} stroke="#6E8FC4" label={best ? undefined : { value: "you", fill: "#6E8FC4", fontSize: 11, position: "insideTopRight" }} />
                  <Area type="monotone" dataKey="net_profit" stroke="#6E8FC4" strokeWidth={2.5} fill="url(#g-profit)" />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <Skeleton className="h-[300px]" />
            )}
          </Card>

          {r && (
            <Card title="Strategy comparison">
              <div className="grid gap-3 sm:grid-cols-3">
                {[
                  { name: "Model-targeted", value: r.net_profit, desc: `Contact ${fmt.compact(r.targeted)} predicted churners`, highlight: true },
                  { name: "Blanket campaign", value: r.target_everyone_profit, desc: `Contact all ${fmt.compact(r.cohort_size)} customers` },
                  { name: "Do nothing", value: 0, desc: "No retention spend, no saves" },
                ].map((s) => (
                  <div key={s.name} className={clsx("rounded-xl border p-4", s.highlight ? "border-accent/50 bg-accent-soft" : "border-line bg-page")}>
                    <div className="text-xs font-semibold uppercase tracking-wider text-ink-soft">{s.name}</div>
                    <div className={clsx("mt-2 font-display text-2xl font-semibold tabular-nums", s.value < 0 ? "text-crit-text" : "text-ink")}>{fmt.moneyCompact(s.value)}</div>
                    <div className="mt-1 text-xs text-ink-faint">{s.desc}</div>
                  </div>
                ))}
              </div>
              <p className="mt-4 text-sm text-ink-soft">
                Targeting with the model is worth{" "}
                <span className={clsx("font-semibold", lift >= 0 ? "text-low-text" : "text-crit-text")}>{fmt.moneyCompact(lift)}</span> more than a blanket campaign
                on this cohort.
              </p>
            </Card>
          )}
        </div>
      </div>
    </>
  );
}
