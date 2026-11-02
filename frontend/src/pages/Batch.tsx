import clsx from "clsx";
import { Download, FileSpreadsheet, FileUp, Loader2, TriangleAlert, X } from "lucide-react";
import { useRef, useState } from "react";
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import { Card, ChartTooltip, KpiCard, PageHeader, RiskBadge } from "../components/ui";
import { api } from "../lib/api";
import { fmt, TIER_STYLES } from "../lib/format";
import type { BatchResponse, RiskTier } from "../lib/types";

const TEMPLATE = `CustomerID,Age,Gender,Tenure,Usage Frequency,Support Calls,Payment Delay,Subscription Type,Contract Length,Total Spend,Last Interaction
1001,34,Male,48,25,1,3,Premium,Annual,920,3
1002,45,Female,20,12,4,14,Standard,Quarterly,610,14
1003,52,Female,6,4,8,26,Basic,Monthly,280,24
1004,29,Male,36,18,2,5,Standard,Annual,780,6
1005,61,Female,12,7,6,19,Basic,Quarterly,410,18
`;

function download(name: string, content: string) {
  const url = URL.createObjectURL(new Blob([content], { type: "text/csv;charset=utf-8" }));
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  a.click();
  URL.revokeObjectURL(url);
}

const csvCell = (v: unknown) => {
  const s = v === null || v === undefined ? "" : String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

export default function Batch() {
  const [file, setFile] = useState<File | null>(null);
  const [drag, setDrag] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [res, setRes] = useState<BatchResponse | null>(null);
  const input = useRef<HTMLInputElement>(null);

  const run = async (f: File) => {
    setFile(f);
    setBusy(true);
    setError(null);
    setRes(null);
    try {
      setRes(await api.predictCsv(f));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const exportResults = () => {
    if (!res) return;
    const header = ["row", "customer_id", "churn_probability", "risk_tier", "will_churn", "segment", "top_driver", "total_spend"];
    const lines = res.results.map((r) => header.map((h) => csvCell(r[h as keyof typeof r])).join(","));
    download(`churnsense-scores-${new Date().toISOString().slice(0, 10)}.csv`, [header.join(","), ...lines].join("\n"));
  };

  const pie = res ? (Object.entries(res.summary.tiers) as [RiskTier, number][]).filter(([, n]) => n > 0) : [];
  const top = res ? [...res.results].sort((a, b) => b.churn_probability - a.churn_probability).slice(0, 50) : [];

  return (
    <>
      <PageHeader
        eyebrow="Bulk scoring"
        title={
          <>
            Score your whole <span className="text-accent-text">customer list</span>
          </>
        }
        subtitle="Upload a CSV export from your CRM. Every row is validated and scored, and you can download the results for your campaign tool."
        actions={
          <button className="btn-ghost" onClick={() => download("churnsense-template.csv", TEMPLATE)}>
            <FileSpreadsheet className="h-4 w-4" /> Template
          </button>
        }
      />

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          const f = e.dataTransfer.files[0];
          if (f) run(f);
        }}
        onClick={() => input.current?.click()}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
        role="button"
        tabIndex={0}
        className={clsx(
          "card flex cursor-pointer flex-col items-center justify-center gap-3 border-2 border-dashed px-6 py-14 text-center transition animate-rise",
          drag ? "border-accent bg-accent-soft" : "border-line hover:border-accent hover:bg-mist",
        )}
      >
        <input
          ref={input}
          type="file"
          accept=".csv,text/csv"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) run(f);
            e.target.value = "";
          }}
        />
        <div className="grid h-14 w-14 place-items-center rounded-2xl bg-accent-soft">
          {busy ? <Loader2 className="h-6 w-6 animate-spin text-ink" /> : <FileUp className="h-6 w-6 text-ink" />}
        </div>
        <div>
          <div className="font-display text-lg font-semibold text-ink">{busy ? `Scoring ${file?.name}…` : "Drop a CSV here or click to browse"}</div>
          <p className="mt-1 text-sm text-ink-soft">Original dataset headers or snake_case both work · up to 10,000 rows / 5 MB</p>
        </div>
      </div>

      {error && (
        <div className="mt-4 flex items-start gap-3 rounded-xl border border-crit/60 bg-crit-soft p-4 text-sm text-crit-text">
          <TriangleAlert className="mt-0.5 h-4 w-4 shrink-0" /> {error}
          <button className="ml-auto" onClick={() => setError(null)} aria-label="Dismiss">
            <X className="h-4 w-4" />
          </button>
        </div>
      )}

      {res && (
        <div className="mt-6 space-y-4">
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <KpiCard label="Rows scored" value={fmt.num(res.summary.scored)} hint={res.summary.failed ? `${res.summary.failed} row${res.summary.failed === 1 ? "" : "s"} failed validation` : "All rows valid"} icon={<FileSpreadsheet className="h-4.5 w-4.5" />} />
            <KpiCard label="Predicted churners" value={fmt.num(res.summary.predicted_churners)} hint={fmt.pct(res.summary.predicted_churners / Math.max(1, res.summary.scored)) + " of scored"} icon={<TriangleAlert className="h-4.5 w-4.5" />} accent="rose" />
            <KpiCard label="Avg churn probability" value={fmt.pct(res.summary.avg_probability)} icon={<FileUp className="h-4.5 w-4.5" />} accent="cyan" />
            <KpiCard label="Spend at risk" value={fmt.moneyCompact(res.summary.revenue_at_risk)} hint="Total spend of predicted churners" icon={<Download className="h-4.5 w-4.5" />} accent="emerald" />
          </div>

          <div className="grid gap-4 lg:grid-cols-[320px_minmax(0,1fr)]">
            <Card title="Risk mix">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={pie.map(([t, n]) => ({ name: TIER_STYLES[t].label, value: n, tier: t }))} dataKey="value" innerRadius={60} outerRadius={90} paddingAngle={3} stroke="none">
                    {pie.map(([t]) => (
                      <Cell key={t} fill={TIER_STYLES[t].color} />
                    ))}
                  </Pie>
                  <Tooltip content={({ active, payload }) => <ChartTooltip active={active} payload={payload?.map((p) => ({ name: String(p.name), value: Number(p.value), color: TIER_STYLES[(p.payload as { tier: RiskTier }).tier].color }))} formatter={fmt.num} />} />
                </PieChart>
              </ResponsiveContainer>
              <div className="mt-2 flex flex-wrap justify-center gap-2">
                {pie.map(([t, n]) => (
                  <span key={t} className="flex items-center gap-1.5 text-xs text-ink-soft">
                    <RiskBadge tier={t} /> {fmt.num(n)}
                  </span>
                ))}
              </div>
            </Card>

            <Card
              title="Highest-risk rows"
              subtitle={`Top ${top.length} of ${fmt.num(res.summary.scored)}`}
              action={
                <button className="btn-primary px-3 py-2 text-xs" onClick={exportResults}>
                  <Download className="h-3.5 w-3.5" /> Download all
                </button>
              }
            >
              <div className="max-h-[360px] overflow-auto">
                <table className="w-full min-w-[560px] text-sm">
                  <thead className="sticky top-0 bg-white">
                    <tr className="text-left text-xs uppercase tracking-wider text-ink-faint">
                      {["Row", "Customer", "Risk", "Segment", "Top driver", "Spend"].map((h) => (
                        <th key={h} className="px-3 py-2 font-medium">
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-line">
                    {top.map((r) => (
                      <tr key={r.row}>
                        <td className="px-3 py-2 text-ink-faint tabular-nums">{r.row + 1}</td>
                        <td className="px-3 py-2 font-mono text-ink-soft">{r.customer_id ?? "—"}</td>
                        <td className="px-3 py-2">
                          <div className="flex items-center gap-2">
                            <span className="w-12 tabular-nums text-ink">{fmt.pct(r.churn_probability, 0)}</span>
                            <RiskBadge tier={r.risk_tier} />
                          </div>
                        </td>
                        <td className="px-3 py-2 text-ink-soft">{r.segment}</td>
                        <td className="px-3 py-2 text-ink-soft">{r.top_driver ?? "—"}</td>
                        <td className="px-3 py-2 tabular-nums text-ink-soft">{fmt.money(r.total_spend)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          </div>

          {res.errors.length > 0 && (
            <Card title={`${res.errors.length} rows skipped`} subtitle="Fix these rows and re-upload">
              <ul className="max-h-48 space-y-1 overflow-auto font-mono text-xs text-med-text">
                {res.errors.slice(0, 100).map((e) => (
                  <li key={e.row}>
                    row {e.row + 1}: {e.error}
                  </li>
                ))}
              </ul>
            </Card>
          )}
        </div>
      )}
    </>
  );
}
