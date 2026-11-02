import { TIER_STYLES, tierOf } from "../lib/format";

/** Semicircular risk gauge, 0-100%. */
export function Gauge({ value, threshold }: { value: number; threshold: number }) {
  const r = 80;
  const c = Math.PI * r;
  const color = TIER_STYLES[tierOf(value)].color;
  const angle = Math.PI * (1 - threshold);
  const tx = 100 + r * Math.cos(angle);
  const ty = 100 - r * Math.sin(angle);
  return (
    <svg viewBox="0 0 200 118" className="w-full max-w-[280px]" role="img" aria-label={`Churn probability ${(value * 100).toFixed(1)}%`}>
      <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="#EEF2F8" strokeWidth="14" strokeLinecap="round" />
      <path
        d="M20 100 A80 80 0 0 1 180 100"
        fill="none"
        stroke={color}
        strokeWidth="14"
        strokeLinecap="round"
        strokeDasharray={c}
        strokeDashoffset={c * (1 - value)}
        style={{ transition: "stroke-dashoffset 0.8s cubic-bezier(.2,.7,.2,1), stroke 0.4s" }}
      />
      <line x1={tx} y1={ty} x2={100 + (r - 18) * Math.cos(angle)} y2={100 - (r - 18) * Math.sin(angle)} stroke="#8590A3" strokeWidth="2" strokeDasharray="2 2" />
      <text x="100" y="88" textAnchor="middle" className="fill-ink font-display" style={{ fontSize: 32, fontWeight: 600 }}>
        {(value * 100).toFixed(1)}%
      </text>
      <text x="100" y="108" textAnchor="middle" style={{ fontSize: 10, fill: "#8590A3", letterSpacing: 2 }}>
        CHURN PROBABILITY
      </text>
    </svg>
  );
}
