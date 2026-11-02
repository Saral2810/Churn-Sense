import { Info } from "lucide-react";
import { Area, AreaChart, Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Card, ChartTooltip, ErrorState, PageHeader, Skeleton } from "../components/ui";
import { useApi } from "../hooks/useApi";
import { api } from "../lib/api";
import { fmt } from "../lib/format";

const pretty = (s: string) => s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

function CurveChart({ data, xLabel, yLabel, diagonal }: { data: { x: number; y: number }[]; xLabel: string; yLabel: string; diagonal?: boolean }) {
  const id = `g-${xLabel.replace(/\W/g, "")}`;
  return (
    <ResponsiveContainer width="100%" height={240}>
      <AreaChart data={data} margin={{ top: 8, right: 8, left: -12, bottom: 12 }}>
        <defs>
          <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="#6E8FC4" stopOpacity={0.4} />
            <stop offset="1" stopColor="#6E8FC4" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid />
        <XAxis dataKey="x" type="number" domain={[0, 1]} tickFormatter={(v) => fmt.dec(v, 1)} axisLine={false} tickLine={false} label={{ value: xLabel, position: "insideBottom", offset: -8, fill: "#8590A3", fontSize: 11 }} />
        <YAxis domain={[0, 1]} tickFormatter={(v) => fmt.dec(v, 1)} axisLine={false} tickLine={false} />
        <Tooltip content={({ active, payload }) => <ChartTooltip active={active} payload={payload?.map((p) => ({ name: yLabel, value: Number(p.value), color: "#6E8FC4" }))} formatter={(v) => fmt.dec(v, 3)} />} />
        {diagonal && <Line data={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} dataKey="y" stroke="#C5CFDF" strokeDasharray="4 4" dot={false} />}
        <Area type="monotone" dataKey="y" stroke="#6E8FC4" strokeWidth={2.5} fill={`url(#${id})`} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

export default function ModelCard() {
  const { data: m, error, reload } = useApi(api.model);
  if (error) return <ErrorState message={error} onRetry={reload} />;
  if (!m)
    return (
      <div className="space-y-4">
        <Skeleton className="h-16 w-80" />
        <Skeleton className="h-40" />
        <Skeleton className="h-72" />
      </div>
    );

  const cm = m.confusion_matrix;
  const total = cm.tn + cm.fp + cm.fn + cm.tp;
  const metrics = [
    ["ROC-AUC", m.holdout.roc_auc],
    ["PR-AUC", m.holdout.pr_auc],
    ["Precision", m.holdout.precision],
    ["Recall", m.holdout.recall],
    ["F1", m.holdout.f1],
    ["Brier ↓", m.holdout.brier],
  ] as const;

  return (
    <>
      <PageHeader
        eyebrow="Transparency"
        title={
          <>
            Model <span className="text-accent-text">card</span>
          </>
        }
        subtitle={`${pretty(m.model_name)} · version ${m.version} · trained ${new Date(m.trained_at).toLocaleString()}`}
      />

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-6">
        {metrics.map(([k, v], i) => (
          <div key={k} className="card px-4 py-4 animate-rise" style={{ animationDelay: `${i * 50}ms` }}>
            <div className="field-label">{k}</div>
            <div className="mt-1 font-display text-2xl font-semibold text-ink tabular-nums">{fmt.dec(v, k.startsWith("Brier") ? 4 : 3)}</div>
          </div>
        ))}
      </div>

      <div className="mt-4 flex items-start gap-3 rounded-xl border border-med/60 bg-med-soft p-4 text-sm text-med-text">
        <Info className="mt-0.5 h-4 w-4 shrink-0 text-med-text" />
        <ul className="space-y-1">
          {m.notes.map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <Card title="ROC curve" subtitle={`AUC ${fmt.dec(m.holdout.roc_auc, 4)}`}>
          <CurveChart data={m.roc_curve} xLabel="False positive rate" yLabel="True positive rate" diagonal />
        </Card>
        <Card title="Precision-recall" subtitle={`Average precision ${fmt.dec(m.holdout.pr_auc, 4)}`}>
          <CurveChart data={m.pr_curve} xLabel="Recall" yLabel="Precision" />
        </Card>
        <Card title="Confusion matrix" subtitle={`At decision threshold ${fmt.pct(m.threshold, 0)} · ${fmt.num(total)} holdout rows`}>
          <div className="grid grid-cols-[auto_1fr_1fr] gap-2 text-center text-xs">
            <div />
            <div className="text-ink-faint">Pred. stay</div>
            <div className="text-ink-faint">Pred. churn</div>
            {(
              [
                ["Stayed", cm.tn, cm.fp, "TN", "FP"],
                ["Churned", cm.fn, cm.tp, "FN", "TP"],
              ] as const
            ).map(([label, a, b, la, lb]) => (
              <div key={label} className="contents">
                <div className="flex items-center pr-1 text-ink-faint [writing-mode:vertical-rl] rotate-180">{label}</div>
                {[
                  [a, la],
                  [b, lb],
                ].map(([v, l]) => {
                  const good = l === "TN" || l === "TP";
                  return (
                    <div
                      key={l}
                      className="rounded-xl py-6 ring-1 ring-line"
                      style={{ background: good ? `rgb(140 197 192 / ${0.08 + (Number(v) / total) * 0.5})` : `rgb(229 156 156 / ${0.05 + (Number(v) / total) * 4})` }}
                    >
                      <div className="font-display text-xl font-semibold text-ink tabular-nums">{fmt.compact(Number(v))}</div>
                      <div className="mt-1 text-[10px] font-semibold tracking-wider text-ink-soft">{l}</div>
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
        </Card>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Card title="Global feature importance" subtitle="Permutation importance: drop in ROC-AUC when a feature is shuffled">
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={m.feature_importance} layout="vertical" margin={{ top: 0, right: 12, left: 8, bottom: 0 }}>
              <defs>
                <linearGradient id="g-imp" x1="0" x2="1">
                  <stop offset="0" stopColor="#B9CCE8" />
                  <stop offset="1" stopColor="#6E8FC4" />
                </linearGradient>
              </defs>
              <XAxis type="number" axisLine={false} tickLine={false} tickFormatter={(v) => fmt.dec(v, 2)} />
              <YAxis type="category" dataKey="label" width={170} axisLine={false} tickLine={false} />
              <Tooltip cursor={{ fill: "rgb(110 143 196 / 0.08)" }} content={({ active, payload, label }) => <ChartTooltip active={active} label={label as string} payload={payload?.map((p) => ({ name: "Δ AUC", value: Number(p.value), color: "#6E8FC4" }))} formatter={(v) => fmt.dec(v, 4)} />} />
              <Bar dataKey="importance" fill="url(#g-imp)" radius={[0, 6, 6, 0]} maxBarSize={18} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
        <Card title="Calibration" subtitle="Predicted probability vs observed churn rate (quantile bins)">
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={m.calibration} margin={{ top: 8, right: 12, left: -12, bottom: 12 }}>
              <CartesianGrid />
              <XAxis dataKey="predicted" type="number" domain={[0, 1]} tickFormatter={(v) => fmt.dec(v, 1)} axisLine={false} tickLine={false} label={{ value: "Predicted", position: "insideBottom", offset: -8, fill: "#8590A3", fontSize: 11 }} />
              <YAxis domain={[0, 1]} tickFormatter={(v) => fmt.dec(v, 1)} axisLine={false} tickLine={false} />
              <Tooltip content={({ active, payload }) => <ChartTooltip active={active} payload={payload?.map((p) => ({ name: "Observed", value: Number(p.value), color: "#8CC5C0" }))} formatter={(v) => fmt.pct(v)} />} />
              <Line data={[{ predicted: 0, observed: 0 }, { predicted: 1, observed: 1 }]} dataKey="observed" stroke="#C5CFDF" strokeDasharray="4 4" dot={false} />
              <Line dataKey="observed" stroke="#8CC5C0" strokeWidth={2.5} dot={{ r: 4, fill: "#8CC5C0" }} />
            </LineChart>
          </ResponsiveContainer>
        </Card>
      </div>

      <Card title="Model comparison" subtitle="All candidates benchmarked on the same holdout split; the winner ships" className="mt-4">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[720px] text-sm">
            <thead>
              <tr className="text-left text-xs uppercase tracking-wider text-ink-faint">
                {["Model", "ROC-AUC", "F1", "Accuracy", "Brier", "Train time", "Latency / row", "Artifact"].map((h) => (
                  <th key={h} className="px-3 py-2 font-medium">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-line">
              {m.comparison.map((c) => (
                <tr key={c.model} className={c.model === m.model_name ? "bg-accent-soft" : ""}>
                  <td className="px-3 py-3 font-medium text-ink">
                    {pretty(c.model)}
                    {c.model === m.model_name && <span className="chip ml-2 bg-accent-soft py-0.5 text-[10px] text-accent-text">deployed</span>}
                  </td>
                  <td className="px-3 py-3 tabular-nums">{fmt.dec(c.roc_auc, 4)}</td>
                  <td className="px-3 py-3 tabular-nums">{fmt.dec(c.f1, 4)}</td>
                  <td className="px-3 py-3 tabular-nums">{fmt.pct(c.accuracy, 2)}</td>
                  <td className="px-3 py-3 tabular-nums">{fmt.dec(c.brier, 4)}</td>
                  <td className="px-3 py-3 tabular-nums">{c.train_seconds}s</td>
                  <td className="px-3 py-3 tabular-nums">{c.inference_us_per_row} µs</td>
                  <td className="px-3 py-3 tabular-nums">{c.artifact_kb >= 1024 ? `${(c.artifact_kb / 1024).toFixed(1)} MB` : `${Math.round(c.artifact_kb)} KB`}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <div className="mt-4 grid gap-4 md:grid-cols-2">
        <Card title="Training data">
          <dl className="grid grid-cols-2 gap-y-3 text-sm">
            {[
              ["Rows", fmt.num(m.data.rows)],
              ["Train / test", `${fmt.num(m.data.train_rows)} / ${fmt.num(m.data.test_rows)}`],
              ["Churn base rate", fmt.pct(m.data.positive_rate)],
              ["Fingerprint", m.data.sha256.slice(0, 16) + "…"],
            ].map(([k, v]) => (
              <div key={k} className="contents">
                <dt className="text-ink-faint">{k}</dt>
                <dd className="font-mono text-ink">{v}</dd>
              </div>
            ))}
          </dl>
        </Card>
        <Card title="Environment">
          <dl className="grid grid-cols-2 gap-y-3 text-sm">
            {Object.entries(m.environment).map(([k, v]) => (
              <div key={k} className="contents">
                <dt className="text-ink-faint">{pretty(k)}</dt>
                <dd className="font-mono text-ink">{v}</dd>
              </div>
            ))}
          </dl>
        </Card>
      </div>
    </>
  );
}
