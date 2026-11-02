import { Loader2, ServerCrash } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { API_URL, api } from "../lib/api";
import type { Health } from "../lib/types";

const MAX_ATTEMPTS = 8;

/**
 * Free hosting tiers (e.g. Render) sleep after inactivity and take ~30-60s to wake.
 * Block the UI with a friendly splash and retry until the API answers.
 */
export function ApiGate({ children }: { children: (health: Health) => ReactNode }) {
  const [health, setHealth] = useState<Health | null>(null);
  const [attempt, setAttempt] = useState(1);
  const [failed, setFailed] = useState(false);
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (health || failed) return;
    let alive = true;
    api
      .health()
      .then((h) => alive && setHealth(h))
      .catch(() => {
        if (!alive) return;
        if (attempt >= MAX_ATTEMPTS) setFailed(true);
        else setTimeout(() => alive && setAttempt((a) => a + 1), 4000);
      });
    return () => {
      alive = false;
    };
  }, [attempt, health, failed]);

  useEffect(() => {
    if (health || failed) return;
    const t = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(t);
  }, [health, failed]);

  if (health) return <>{children(health)}</>;

  return (
    <div className="grid min-h-screen place-items-center px-4">
      <div className="card card-pad w-full max-w-md text-center animate-rise">
        <img src="/favicon.svg" alt="" className="mx-auto h-14 w-14 rounded-2xl ring-1 ring-line" />
        {failed ? (
          <>
            <ServerCrash className="mx-auto mt-6 h-8 w-8 text-crit-text" />
            <h1 className="mt-3 font-display text-xl font-semibold text-ink">Can't reach the API</h1>
            <p className="mt-2 text-sm text-ink-soft">
              Tried <span className="font-mono text-ink-soft">{API_URL}</span> {MAX_ATTEMPTS} times. Check that the backend is
              running and that <span className="font-mono">CORS_ORIGINS</span> includes this site.
            </p>
            <button
              className="btn-primary mt-6"
              onClick={() => {
                setFailed(false);
                setAttempt(1);
                setElapsed(0);
              }}
            >
              Retry
            </button>
          </>
        ) : (
          <>
            <h1 className="mt-6 font-display text-xl font-semibold text-ink">
              Waking up <span className="text-accent-text">ChurnSense</span>
            </h1>
            <p className="mt-2 text-sm text-ink-soft">
              {elapsed < 5
                ? "Connecting to the prediction engine…"
                : "The free-tier server is starting up. This can take up to a minute on the first visit."}
            </p>
            <div className="mt-6 h-1.5 overflow-hidden rounded-full bg-mist">
              <div
                className="h-full rounded-full bg-accent transition-[width] duration-1000"
                style={{ width: `${Math.min(95, 8 + elapsed * 1.6)}%` }}
              />
            </div>
            <div className="mt-3 flex items-center justify-center gap-2 text-xs text-ink-faint">
              <Loader2 className="h-3.5 w-3.5 animate-spin" /> {elapsed}s · attempt {attempt}/{MAX_ATTEMPTS}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
