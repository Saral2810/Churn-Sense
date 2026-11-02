import clsx from "clsx";
import { AlertTriangle, RefreshCw } from "lucide-react";
import type { ReactNode } from "react";
import { TIER_STYLES } from "../lib/format";
import type { RiskTier } from "../lib/types";

export function PageHeader({ eyebrow, title, subtitle, actions }: { eyebrow: string; title: ReactNode; subtitle?: string; actions?: ReactNode }) {
  return (
    <header className="mb-6 flex flex-col gap-4 sm:mb-8 sm:flex-row sm:items-end sm:justify-between animate-rise">
      <div>
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-accent-text">{eyebrow}</p>
        <h1 className="mt-2 font-display text-3xl font-semibold tracking-tight text-ink sm:text-4xl">{title}</h1>
        {subtitle && <p className="mt-2 max-w-2xl text-sm text-ink-soft sm:text-base">{subtitle}</p>}
      </div>
      {actions && <div className="flex shrink-0 gap-2">{actions}</div>}
    </header>
  );
}

export function Card({ title, subtitle, action, children, className }: { title?: string; subtitle?: string; action?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={clsx("card card-pad animate-rise", className)}>
      {(title || action) && (
        <div className="mb-4 flex items-start justify-between gap-3">
          <div>
            {title && <h2 className="font-display text-base font-semibold text-ink">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-xs text-ink-soft">{subtitle}</p>}
          </div>
          {action}
        </div>
      )}
      {children}
    </section>
  );
}

export function KpiCard({ label, value, hint, icon, accent = "brand", delay = 0 }: { label: string; value: ReactNode; hint?: ReactNode; icon: ReactNode; accent?: "brand" | "cyan" | "rose" | "emerald"; delay?: number }) {
  const accents = {
    brand: "bg-accent-soft text-accent-text",
    cyan: "bg-accent-soft text-accent-text",
    rose: "bg-crit-soft text-crit-text",
    emerald: "bg-low-soft text-low-text",
  };
  return (
    <div className="card card-pad animate-rise" style={{ animationDelay: `${delay}ms` }}>
      <div className="flex items-center justify-between">
        <span className="field-label">{label}</span>
        <span className={clsx("grid h-9 w-9 place-items-center rounded-xl", accents[accent])}>{icon}</span>
      </div>
      <div className="mt-3 font-display text-3xl font-semibold tracking-tight text-ink tabular-nums">{value}</div>
      {hint && <div className="mt-1 text-xs text-ink-soft">{hint}</div>}
    </div>
  );
}

export function RiskBadge({ tier }: { tier: RiskTier }) {
  const s = TIER_STYLES[tier];
  return (
    <span className={clsx("chip", s.chip)}>
      <span className={clsx("h-1.5 w-1.5 rounded-full", s.dot)} />
      {s.label}
    </span>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={clsx("animate-pulse rounded-xl bg-mist", className)} />;
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="card card-pad flex flex-col items-center gap-3 py-12 text-center">
      <AlertTriangle className="h-8 w-8 text-med-text" />
      <p className="text-sm text-ink-soft">{message}</p>
      {onRetry && (
        <button className="btn-ghost" onClick={onRetry}>
          <RefreshCw className="h-4 w-4" /> Try again
        </button>
      )}
    </div>
  );
}

export function Segmented<T extends string>({ value, options, onChange }: { value: T; options: { value: T; label: string }[]; onChange: (v: T) => void }) {
  return (
    <div className="inline-flex rounded-xl border border-line bg-page p-1" role="radiogroup">
      {options.map((o) => (
        <button
          key={o.value}
          role="radio"
          aria-checked={value === o.value}
          onClick={() => onChange(o.value)}
          className={clsx(
            "rounded-lg px-3 py-1.5 text-xs font-semibold transition",
            value === o.value ? "bg-white text-accent-text shadow-sm ring-1 ring-line" : "text-ink-soft hover:text-ink",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

interface TooltipPayload {
  name?: string;
  value?: number;
  color?: string;
  payload?: Record<string, unknown>;
}

export function ChartTooltip({
  active,
  payload,
  label,
  formatter,
}: {
  active?: boolean;
  payload?: TooltipPayload[];
  label?: string | number;
  formatter?: (v: number, name?: string) => string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-xl border border-line bg-white px-3 py-2 text-xs shadow-sm">
      {label !== undefined && <div className="mb-1 font-semibold text-ink">{label}</div>}
      {payload.map((p, i) => (
        <div key={i} className="flex items-center gap-2 text-ink-soft">
          <span className="h-2 w-2 rounded-full" style={{ background: p.color }} />
          <span>{p.name}</span>
          <span className="ml-auto pl-3 font-semibold text-ink tabular-nums">
            {formatter ? formatter(Number(p.value), p.name) : p.value}
          </span>
        </div>
      ))}
    </div>
  );
}
