import clsx from "clsx";
import { ChevronLeft, ChevronRight, Search, Wand2 } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { ErrorState, PageHeader, RiskBadge, Segmented, Skeleton } from "../components/ui";
import { useApi, useDebounced } from "../hooks/useApi";
import { api } from "../lib/api";
import { fmt, TIER_STYLES } from "../lib/format";
import type { CustomerInput, RiskTier, ScoredCustomer } from "../lib/types";

const PAGE = 15;
type TierFilter = "" | RiskTier;

function toInput(c: ScoredCustomer): CustomerInput {
  const { age, gender, tenure, usage_frequency, support_calls, payment_delay, subscription_type, contract_length, total_spend, last_interaction } = c;
  return { customer_id: String(c.customer_id), age, gender, tenure, usage_frequency, support_calls, payment_delay, subscription_type, contract_length, total_spend, last_interaction };
}

export default function Customers() {
  const [tier, setTier] = useState<TierFilter>("");
  const [sort, setSort] = useState("risk");
  const [q, setQ] = useState("");
  const [page, setPage] = useState(0);
  const query = useDebounced(q, 300);
  const navigate = useNavigate();

  const { data, error, loading, reload } = useApi(
    () => api.customers({ tier: tier || undefined, q: query || undefined, sort, limit: PAGE, offset: page * PAGE }),
    [tier, query, sort, page],
  );

  const pages = data ? Math.max(1, Math.ceil(data.total / PAGE)) : 1;
  const counts = data?.tiers ?? {};
  const all = Object.values(counts).reduce((s, n) => s + (n ?? 0), 0);

  return (
    <>
      <PageHeader
        eyebrow="Watchlist"
        title={
          <>
            At-risk <span className="text-accent-text">customers</span>
          </>
        }
        subtitle="A scored sample of holdout customers the model never saw in training. Pick anyone to see why they're at risk."
      />

      <div className="mb-4 grid gap-3 sm:grid-cols-4">
        {(["critical", "high", "medium", "low"] as const).map((t) => (
          <button
            key={t}
            onClick={() => {
              setTier(tier === t ? "" : t);
              setPage(0);
            }}
            className={clsx("card px-4 py-3 text-left transition hover:bg-mist", tier === t && "ring-2 ring-accent/50")}
          >
            <div className="flex items-center justify-between">
              <RiskBadge tier={t} />
              <span className="text-xs text-ink-faint">{all ? fmt.pct((counts[t] ?? 0) / all, 0) : "…"}</span>
            </div>
            <div className="mt-2 font-display text-2xl font-semibold text-ink tabular-nums">{fmt.num(counts[t] ?? 0)}</div>
          </button>
        ))}
      </div>

      <div className="card overflow-hidden animate-rise">
        <div className="flex flex-col gap-3 border-b border-line p-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="relative sm:w-64">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-ink-faint" />
            <input
              className="field pl-9"
              placeholder="Search customer ID"
              value={q}
              inputMode="numeric"
              onChange={(e) => {
                setQ(e.target.value.replace(/\D/g, ""));
                setPage(0);
              }}
            />
          </div>
          <Segmented
            value={sort}
            onChange={(v) => {
              setSort(v);
              setPage(0);
            }}
            options={[
              { value: "risk", label: "Highest risk" },
              { value: "value_at_risk", label: "Value at risk" },
              { value: "spend", label: "Top spend" },
            ]}
          />
        </div>

        {error ? (
          <div className="p-4">
            <ErrorState message={error} onRetry={reload} />
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[860px] text-sm">
              <thead>
                <tr className="text-left text-xs uppercase tracking-wider text-ink-faint">
                  {["Customer", "Churn risk", "Segment", "Contract", "Tenure", "Support", "Pay delay", "Spend", "Outcome", ""].map((h) => (
                    <th key={h} className="px-4 py-3 font-medium">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className={clsx("divide-y divide-line transition-opacity", loading && "opacity-60")}>
                {!data
                  ? Array.from({ length: 8 }, (_, i) => (
                      <tr key={i}>
                        <td colSpan={10} className="px-4 py-2">
                          <Skeleton className="h-8" />
                        </td>
                      </tr>
                    ))
                  : data.items.map((c) => (
                      <tr key={c.customer_id} className="group hover:bg-page">
                        <td className="px-4 py-3 font-mono text-ink-soft">#{c.customer_id}</td>
                        <td className="px-4 py-3">
                          <div className="flex items-center gap-3">
                            <div className="h-1.5 w-20 overflow-hidden rounded-full bg-mist">
                              <div className="h-full rounded-full" style={{ width: fmt.pct(c.churn_probability), background: TIER_STYLES[c.risk_tier].color }} />
                            </div>
                            <span className="w-12 tabular-nums text-ink">{fmt.pct(c.churn_probability, 0)}</span>
                            <RiskBadge tier={c.risk_tier} />
                          </div>
                        </td>
                        <td className="px-4 py-3 text-ink-soft">{c.segment}</td>
                        <td className="px-4 py-3 text-ink-soft">{c.contract_length}</td>
                        <td className="px-4 py-3 tabular-nums text-ink-soft">{c.tenure}m</td>
                        <td className="px-4 py-3 tabular-nums text-ink-soft">{c.support_calls}</td>
                        <td className="px-4 py-3 tabular-nums text-ink-soft">{c.payment_delay}d</td>
                        <td className="px-4 py-3 tabular-nums text-ink">{fmt.money(c.total_spend)}</td>
                        <td className="px-4 py-3">
                          <span className={clsx("text-xs", c.actual_churn ? "text-crit-text" : "text-low-text")}>{c.actual_churn ? "Churned" : "Stayed"}</span>
                        </td>
                        <td className="px-4 py-3 text-right">
                          <button
                            className="btn-ghost px-2.5 py-1.5 text-xs opacity-80 group-hover:opacity-100"
                            onClick={() => navigate("/predict", { state: { customer: toInput(c) } })}
                          >
                            <Wand2 className="h-3.5 w-3.5" /> Analyze
                          </button>
                        </td>
                      </tr>
                    ))}
                {data && data.items.length === 0 && (
                  <tr>
                    <td colSpan={10} className="px-4 py-12 text-center text-ink-faint">
                      No customers match these filters.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}

        <div className="flex items-center justify-between border-t border-line px-4 py-3 text-xs text-ink-soft">
          <span>{data ? `${fmt.num(data.total)} customers` : "…"}</span>
          <div className="flex items-center gap-2">
            <button className="btn-ghost p-1.5" disabled={page === 0} onClick={() => setPage((p) => p - 1)} aria-label="Previous page">
              <ChevronLeft className="h-4 w-4" />
            </button>
            <span className="tabular-nums">
              {page + 1} / {pages}
            </span>
            <button className="btn-ghost p-1.5" disabled={page + 1 >= pages} onClick={() => setPage((p) => p + 1)} aria-label="Next page">
              <ChevronRight className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>
    </>
  );
}
