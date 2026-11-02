import { AlertOctagon, ArrowRight, BrainCircuit, DollarSign, TrendingDown, Users } from "lucide-react";
import { Area, AreaChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Link } from "react-router-dom";
import { RateBars } from "../components/charts";
import { Card, ChartTooltip, ErrorState, KpiCard, PageHeader, Skeleton } from "../components/ui";
import { useApi } from "../hooks/useApi";
import { api } from "../lib/api";
import { fmt, rateColor } from "../lib/format";

const SEVERITY = {
  critical: "border-crit/60 bg-crit-soft text-crit-text",
  high: "border-high/60 bg-high-soft text-high-text",
  medium: "border-med/60 bg-med-soft text-med-text",
};

export default function Overview() {
  const analytics = useApi(api.analytics);
  const model = useApi(api.model);

  if (analytics.error) return <ErrorState message={analytics.error} onRetry={analytics.reload} />;
  const a = analytics.data;

  return (
    <>
      <PageHeader
        eyebrow="Executive overview"
        title={
          <>
            Know who's leaving, <span className="text-accent-text">before they leave.</span>
          </>
        }
        subtitle="Churn drivers, high-risk segments and revenue exposure across your customer base, powered by a gradient-boosted model."
        actions={
          <Link to="/predict" className="btn-primary">
            Score a customer <ArrowRight className="h-4 w-4" />
          </Link>
        }
      />

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {a ? (
          <>
            <KpiCard label="Customers analysed" value={fmt.compact(a.kpis.customers)} hint={`${fmt.num(a.kpis.churned)} churned`} icon={<Users className="h-4.5 w-4.5" />} />
            <KpiCard label="Churn rate" value={fmt.pct(a.kpis.churn_rate)} hint={`Avg tenure ${a.kpis.avg_tenure} months`} icon={<TrendingDown className="h-4.5 w-4.5" />} accent="rose" delay={60} />
            <KpiCard label="Revenue at risk" value={fmt.moneyCompact(a.kpis.revenue_at_risk)} hint={`Avg spend ${fmt.money(a.kpis.avg_spend)}`} icon={<DollarSign className="h-4.5 w-4.5" />} accent="cyan" delay={120} />
            <KpiCard
              label="Model ROC-AUC"
              value={model.data ? fmt.dec(model.data.holdout.roc_auc, 3) : "…"}
              hint={model.data ? `Recall ${fmt.pct(model.data.holdout.recall)} on holdout` : "Loading model card"}
              icon={<BrainCircuit className="h-4.5 w-4.5" />}
              accent="emerald"
              delay={180}
            />
          </>
        ) : (
          Array.from({ length: 4 }, (_, i) => <Skeleton key={i} className="h-32" />)
        )}
      </div>

      {a && (
        <>
          <div className="mt-6 grid gap-4 lg:grid-cols-3">
            {a.insights.slice(0, 3).map((ins, i) => (
              <div key={ins.title} className={`card card-pad border animate-rise ${SEVERITY[ins.severity]}`} style={{ animationDelay: `${i * 80}ms` }}>
                <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider">
                  <AlertOctagon className="h-4 w-4" /> {ins.severity} insight
                </div>
                <h3 className="mt-2 font-display text-lg font-semibold text-ink">{ins.title}</h3>
                <p className="mt-1.5 text-sm text-ink-soft">{ins.detail}</p>
              </div>
            ))}
          </div>

          <div className="mt-6 grid gap-4 lg:grid-cols-2">
            <Card title="Churn by contract length" subtitle="Monthly plans are the single biggest risk factor">
              <RateBars data={a.by_contract} />
            </Card>
            <Card title="Churn by support calls" subtitle="A sharp cliff above 5 calls">
              <RateBars data={a.by_support_calls} />
            </Card>
            <Card title="Churn by payment delay" subtitle="Days late on the last invoice">
              <RateBars data={a.by_payment_delay} />
            </Card>
            <Card title="Churn by age band">
              <RateBars data={a.by_age} />
            </Card>
          </div>

          <div className="mt-4 grid gap-4 lg:grid-cols-3">
            <Card title="Spend distribution" subtitle="Churned vs retained customers by total spend" className="lg:col-span-2">
              <ResponsiveContainer width="100%" height={260}>
                <AreaChart data={a.spend_distribution} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
                  <defs>
                    <linearGradient id="g-churn" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0" stopColor="#E59C9C" stopOpacity={0.5} />
                      <stop offset="1" stopColor="#E59C9C" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="g-ret" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0" stopColor="#6E8FC4" stopOpacity={0.45} />
                      <stop offset="1" stopColor="#6E8FC4" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid vertical={false} />
                  <XAxis dataKey="bucket" tickFormatter={(v) => `$${v}`} axisLine={false} tickLine={false} />
                  <YAxis tickFormatter={(v) => fmt.compact(v)} axisLine={false} tickLine={false} />
                  <Tooltip content={({ active, payload, label }) => <ChartTooltip active={active} payload={payload as never} label={`Spend ≥ $${label}`} formatter={fmt.num} />} />
                  <Legend iconType="circle" wrapperStyle={{ fontSize: 12 }} />
                  <Area type="monotone" dataKey="retained" name="Retained" stroke="#6E8FC4" strokeWidth={2} fill="url(#g-ret)" />
                  <Area type="monotone" dataKey="churned" name="Churned" stroke="#E59C9C" strokeWidth={2} fill="url(#g-churn)" />
                </AreaChart>
              </ResponsiveContainer>
            </Card>
            <Card title="Subscription tier">
              <RateBars data={a.by_subscription} layout="horizontal" height={260} />
            </Card>
          </div>

          <h2 className="mb-4 mt-10 font-display text-xl font-semibold text-ink">Customer segments</h2>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            {a.segments.map((s, i) => (
              <div key={s.name} className="card card-pad animate-rise" style={{ animationDelay: `${i * 70}ms` }}>
                <div className="flex items-center justify-between">
                  <h3 className="font-display font-semibold text-ink">{s.name}</h3>
                  <span className="chip bg-mist text-ink-soft">{fmt.pct(s.share, 0)}</span>
                </div>
                <div className="mt-4 flex items-end gap-2">
                  <span className="font-display text-3xl font-semibold tabular-nums" style={{ color: rateColor(s.churn_rate) }}>
                    {fmt.pct(s.churn_rate, 0)}
                  </span>
                  <span className="pb-1 text-xs text-ink-soft">churn</span>
                </div>
                <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-mist">
                  <div className="h-full rounded-full" style={{ width: fmt.pct(s.churn_rate), background: rateColor(s.churn_rate) }} />
                </div>
                <dl className="mt-4 grid grid-cols-3 gap-2 text-xs">
                  {[
                    ["Spend", fmt.money(s.avg_spend)],
                    ["Tenure", `${s.avg_tenure}m`],
                    ["Usage", `${s.avg_usage}/mo`],
                  ].map(([k, v]) => (
                    <div key={k}>
                      <dt className="text-ink-faint">{k}</dt>
                      <dd className="font-semibold text-ink">{v}</dd>
                    </div>
                  ))}
                </dl>
              </div>
            ))}
          </div>
        </>
      )}
    </>
  );
}
