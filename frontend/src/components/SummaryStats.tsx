import { Gauge, Target, MapPin, Activity } from "lucide-react";

/**
 * Match Summary — computed ONLY from the deliveries this page was handed,
 * which are the current analysis's own. Nothing is read from storage, from a
 * previous analysis or from a constant: with three measured speeds out of
 * thirty-seven, the average is those three and the other thirty-four simply
 * have no speed. Absent measurements render "—", never a placeholder label.
 */
interface Delivery {
  ball: number;
  speed?: number;
  length?: string;
  line?: string;
  swing?: string;
}
interface Props { deliveries: Delivery[]; processingTime?: number; }

const DASH = "—";

/** Most common of the values that EXIST; "—" when none does. */
const mostCommon = (arr: (string | undefined)[]) => {
  const values = arr.filter((v): v is string => Boolean(v));
  if (!values.length) return DASH;
  const freq: Record<string, number> = {};
  values.forEach((v) => (freq[v] = (freq[v] || 0) + 1));
  return Object.entries(freq).sort((a, b) => b[1] - a[1])[0][0];
};

/** Measured speeds only — a missing speed is not a zero-speed delivery. */
const measuredSpeeds = (deliveries: Delivery[]): number[] =>
  deliveries
    .map((d) => d.speed)
    .filter((s): s is number => typeof s === "number" && Number.isFinite(s) && s > 0);

const SummaryStats = ({ deliveries, processingTime }: Props) => {
  if (!deliveries.length) return null;

  const total       = deliveries.length;
  const speeds      = measuredSpeeds(deliveries);
  const avgSpeed    = speeds.length ? speeds.reduce((s, v) => s + v, 0) / speeds.length : null;
  const maxSpeed    = speeds.length ? Math.max(...speeds) : null;
  const commonLen   = mostCommon(deliveries.map((d) => d.length));
  const commonLine  = mostCommon(deliveries.map((d) => d.line));

  // Breakdowns count every delivery, with the unmeasured ones gathered under
  // "—", so the bars always account for all `total` balls and a missing
  // measurement is visible rather than silently dropped.
  const lenCounts: Record<string, number>  = {};
  const lineCounts: Record<string, number> = {};
  deliveries.forEach((d) => {
    const len = d.length || DASH;
    const line = d.line || DASH;
    lenCounts[len]  = (lenCounts[len]  || 0) + 1;
    lineCounts[line] = (lineCounts[line] || 0) + 1;
  });

  const cards = [
    { icon: Activity, label: "Total Deliveries", value: String(total),
      sub: processingTime ? `Processed in ${processingTime}s` : "", color: "from-blue-500 to-blue-600" },
    { icon: Gauge,    label: "Avg Speed",
      value: avgSpeed !== null ? `${avgSpeed.toFixed(1)} km/h` : DASH,
      sub: maxSpeed !== null ? `Max: ${maxSpeed.toFixed(1)} km/h` : "", color: "from-orange-500 to-red-500" },
    { icon: MapPin,   label: "Top Length", value: commonLen,
      sub: `${lenCounts[commonLen] || 0} of ${total} balls`, color: "from-green-500 to-emerald-600" },
    { icon: Target,   label: "Top Line",   value: commonLine,
      sub: `${lineCounts[commonLine] || 0} of ${total} balls`, color: "from-purple-500 to-violet-600" },
  ];

  return (
    <div className="space-y-4">
      <h2 className="text-2xl md:text-3xl font-bold tracking-tight">
        Match <span className="text-gradient">Summary</span>
      </h2>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {cards.map((c) => (
          <div key={c.label}
            className="rounded-2xl gradient-card border border-border shadow-soft p-5 flex flex-col gap-3 hover:shadow-elegant transition-smooth hover:-translate-y-1">
            <div className={`h-10 w-10 rounded-xl bg-gradient-to-br ${c.color} flex items-center justify-center`}>
              <c.icon className="h-5 w-5 text-white" />
            </div>
            <div>
              <p className="text-xs text-muted-foreground font-medium mb-1">{c.label}</p>
              <p className="text-xl font-bold tracking-tight"
                 data-testid={`summary-${c.label.toLowerCase().replace(/\s+/g, "-")}`}>
                {c.value}
              </p>
              {c.sub && <p className="text-xs text-muted-foreground mt-0.5">{c.sub}</p>}
            </div>
          </div>
        ))}
      </div>

      {/* Length breakdown */}
      <div className="rounded-2xl gradient-card border border-border shadow-soft p-5 space-y-3">
        <p className="text-sm font-semibold">Delivery Length Breakdown</p>
        <div className="space-y-2">
          {Object.entries(lenCounts).sort((a, b) => b[1] - a[1]).map(([len, cnt]) => {
            const pct = Math.round((cnt / total) * 100);
            return (
              <div key={len} className="space-y-1">
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>{len}</span><span>{cnt} balls ({pct}%)</span>
                </div>
                <div className="h-2 w-full rounded-full bg-secondary overflow-hidden">
                  <div className="h-full gradient-primary rounded-full transition-all duration-700"
                    style={{ width: `${pct}%` }} />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default SummaryStats;
