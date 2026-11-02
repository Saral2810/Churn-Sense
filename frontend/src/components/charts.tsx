import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { fmt, rateColor } from "../lib/format";
import type { GroupStat } from "../lib/types";
import { ChartTooltip } from "./ui";

/** Bar chart of churn rate per group, colour-coded by severity. */
export function RateBars({ data, height = 220, layout = "vertical" }: { data: GroupStat[]; height?: number; layout?: "vertical" | "horizontal" }) {
  const horizontal = layout === "horizontal";
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} layout={horizontal ? "vertical" : "horizontal"} margin={{ top: 4, right: 8, left: horizontal ? 8 : -16, bottom: 0 }}>
        {horizontal ? (
          <>
            <XAxis type="number" domain={[0, 1]} tickFormatter={(v) => fmt.pct(v, 0)} axisLine={false} tickLine={false} />
            <YAxis type="category" dataKey="name" width={78} axisLine={false} tickLine={false} />
          </>
        ) : (
          <>
            <XAxis dataKey="name" axisLine={false} tickLine={false} />
            <YAxis domain={[0, 1]} tickFormatter={(v) => fmt.pct(v, 0)} axisLine={false} tickLine={false} />
          </>
        )}
        <Tooltip
          cursor={{ fill: "rgb(110 143 196 / 0.08)" }}
          content={({ active, payload, label }) => (
            <ChartTooltip
              active={active}
              label={label as string}
              payload={
                payload?.length
                  ? [
                      { name: "Churn rate", value: (payload[0].payload as GroupStat).churn_rate, color: rateColor((payload[0].payload as GroupStat).churn_rate) },
                      { name: "Customers", value: (payload[0].payload as GroupStat).customers, color: "#8590A3" },
                    ]
                  : []
              }
              formatter={(v, n) => (n === "Churn rate" ? fmt.pct(v) : fmt.num(v))}
            />
          )}
        />
        <Bar dataKey="churn_rate" radius={horizontal ? [0, 6, 6, 0] : [6, 6, 0, 0]} maxBarSize={44}>
          {data.map((d) => (
            <Cell key={d.name} fill={rateColor(d.churn_rate)} fillOpacity={0.85} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
