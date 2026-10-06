import { Gauge, Target, MapPin, Activity } from "lucide-react";

interface Delivery {
  ball: number; speed: number; length: string; line: string; swing: string;
}
interface Props { deliveries: Delivery[]; processingTime?: number; }

const mostCommon = (arr: string[]) => {
  if (!arr.length) return "—";
  const freq: Record<string, number> = {};
  arr.forEach((v) => (freq[v] = (freq[v] || 0) + 1));
  return Object.entries(freq).sort((a, b) => b[1] - a[1])[0][0];
};

const SummaryStats = ({ deliveries, processingTime }: Props) => {
  if (!deliveries.length) return null;

  const total       = deliveries.length;
  const speedBalls  = deliveries.filter((d) => d.speed > 0);
  const avgSpeed    = speedBalls.length ? Math.round(speedBalls.reduce((s, d) => s + d.speed, 0) / speedBalls.length) : 0;
  const maxSpeed    = speedBalls.length ? Math.max(...speedBalls.map((d) => d.speed)) : 0;
  const commonLen   = mostCommon(deliveries.map((d) => d.length));
  const commonLine  = mostCommon(deliveries.map((d) => d.line));

  const lenCounts: Record<string, number>  = {};
  const lineCounts: Record<string, number> = {};
  deliveries.forEach((d) => {
    lenCounts[d.length]  = (lenCounts[d.length]  || 0) + 1;
    lineCounts[d.line]   = (lineCounts[d.line]   || 0) + 1;
  });

  const cards = [
    { icon: Activity, label: "Total Deliveries", value: String(total),
      sub: processingTime ? `Processed in ${processingTime}s` : "", color: "from-blue-500 to-blue-600" },
    { icon: Gauge,    label: "Avg Speed",  value: avgSpeed ? `${avgSpeed} km/h` : "—",
      sub: maxSpeed ? `Max: ${maxSpeed} km/h` : "", color: "from-orange-500 to-red-500" },
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
              <p className="text-xl font-bold tracking-tight">{c.value}</p>
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
