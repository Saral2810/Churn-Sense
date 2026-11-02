import clsx from "clsx";
import { Activity, BrainCircuit, Calculator, FileUp, Gauge, LayoutDashboard, Users } from "lucide-react";
import { NavLink, Outlet } from "react-router-dom";
import type { Health } from "../lib/types";

const NAV = [
  { to: "/", label: "Overview", icon: LayoutDashboard, end: true },
  { to: "/predict", label: "Live Predictor", icon: Gauge },
  { to: "/customers", label: "At-Risk Customers", icon: Users },
  { to: "/roi", label: "ROI Simulator", icon: Calculator },
  { to: "/batch", label: "Batch Scoring", icon: FileUp },
  { to: "/model", label: "Model Card", icon: BrainCircuit },
];

function Logo() {
  return (
    <div className="flex items-center gap-2.5">
      <img src="/favicon.svg" alt="" className="h-9 w-9 rounded-xl ring-1 ring-line" />
      <div className="leading-tight">
        <div className="font-display text-lg font-semibold tracking-tight text-ink">
          Churn<span className="text-accent-text">Sense</span>
        </div>
        <div className="text-[10px] font-medium uppercase tracking-[0.18em] text-ink-faint">Retention Intelligence</div>
      </div>
    </div>
  );
}

export function Layout({ health }: { health: Health }) {
  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[260px_minmax(0,1fr)]">
      {/* Desktop sidebar */}
      <aside className="sticky top-0 hidden h-screen flex-col border-r border-line bg-white px-4 py-6 lg:flex">
        <div className="px-2">
          <Logo />
        </div>
        <nav className="mt-10 flex flex-col gap-1" aria-label="Main">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) =>
                clsx(
                  "group relative flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition",
                  isActive ? "bg-accent-soft text-accent-text" : "text-ink-soft hover:bg-page hover:text-ink",
                )
              }
            >
              {({ isActive }) => (
                <>
                  {isActive && <span className="absolute inset-y-2 left-0 w-1 rounded-full bg-accent" />}
                  <Icon className={clsx("h-4.5 w-4.5", isActive ? "text-accent-text" : "text-ink-faint group-hover:text-ink-soft")} />
                  {label}
                </>
              )}
            </NavLink>
          ))}
        </nav>
        <div className="mt-auto rounded-2xl border border-line bg-page p-4">
          <div className="flex items-center gap-2 text-xs font-semibold text-low-text">
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-low opacity-60" />
              <span className="relative inline-flex h-2 w-2 rounded-full bg-low" />
            </span>
            API online
          </div>
          <div className="mt-2 text-[11px] leading-relaxed text-ink-faint">
            Model <span className="font-mono text-ink-soft">{health.model_version}</span>
          </div>
        </div>
      </aside>

      {/* Mobile top bar */}
      <div className="sticky top-0 z-20 border-b border-line bg-white/90 lg:hidden">
        <div className="flex items-center justify-between px-4 py-3">
          <Logo />
          <Activity className="h-4 w-4 text-low-text" aria-label="API online" />
        </div>
        <nav className="flex gap-1 overflow-x-auto px-3 pb-3" aria-label="Main">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) =>
                clsx(
                  "flex shrink-0 items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium",
                  isActive ? "bg-accent-soft text-ink ring-1 ring-accent/40" : "text-ink-soft",
                )
              }
            >
              <Icon className="h-3.5 w-3.5" />
              {label}
            </NavLink>
          ))}
        </nav>
      </div>

      <main className="mx-auto w-full max-w-7xl px-4 py-6 sm:px-6 lg:px-10 lg:py-10">
        <Outlet />
        <footer className="mt-12 border-t border-line pt-6 text-xs text-ink-faint">
          ChurnSense · {health.model_name.replace(/_/g, " ")} · trained {new Date(health.trained_at).toLocaleDateString()}
        </footer>
      </main>
    </div>
  );
}
