import { Gauge, Target, MapPin, Wind, Crosshair, Sparkles, RotateCcw, TrendingDown } from "lucide-react";

export interface BallData {
  ball_id: number;
  speed?: number;
  line?: string;
  length?: string;
  swing?: string;
  /** Shot classification, e.g. "Cover"; absent when the model said nothing. */
  shot?: string;
  /** The model's confidence for `shot`, 0..1. */
  shot_confidence?: number;
  release_angle?: number;
  bounce_angle?: number;
  [key: string]: unknown;
}

// Length categories + Unknown fallback. The first five are the metre zones the
// backend publishes (core/constants.py LENGTH_ZONES_M); the last two are kept
// for results cached before length was measured in metres.
const LENGTH_COLOR: Record<string, string> = {
  "Very Short":  "from-violet-500 to-purple-600",
  Short:         "from-blue-500 to-cyan-500",
  "Good Length": "from-green-500 to-emerald-600",
  Full:          "from-orange-500 to-amber-500",
  Yorker:        "from-red-500 to-rose-600",
  Beamer:        "from-purple-600 to-pink-600",
  Bouncer:       "from-pink-500 to-red-500",
  Unknown:       "from-gray-500 to-gray-600",
};

const BallAnalysisCard = ({ data }: { data: BallData }) => {
  const lengthColor = LENGTH_COLOR[data.length as string] ?? LENGTH_COLOR.Unknown;

  // The shot the model actually said, with its confidence rather than hidden:
  // a 23% "Cover" is worth showing as 23%, not as a fact.
  const shotLabel = data.shot
    ? typeof data.shot_confidence === "number"
      ? `${data.shot} (${Math.round(data.shot_confidence * 100)}%)`
      : String(data.shot)
    : null;

  const stats = [
    { key: "speed",  label: "Speed",  value: typeof data.speed === "number" ? `${data.speed.toFixed(1)} km/h` : "—", Icon: Gauge },
    { key: "line",   label: "Line",   value: (data.line   as string) || "—", Icon: Target },
    { key: "length", label: "Length", value: (data.length as string) || "—", Icon: MapPin },
    { key: "swing",  label: "Swing",  value: (data.swing  as string) || "—", Icon: Wind  },
    { key: "shot",   label: "Shot",   value: shotLabel || "—", Icon: Crosshair },
  ];

  // Extra details row (angles) — only if available
  const extras = [
    data.release_angle != null ? { label: "Release", value: `${data.release_angle}°`, Icon: TrendingDown } : null,
    data.bounce_angle  != null ? { label: "Bounce",  value: `${data.bounce_angle}°`,  Icon: RotateCcw   } : null,
  ].filter(Boolean) as { label: string; value: string; Icon: React.ComponentType<{ className?: string }> }[];

  return (
    <div data-ball-id={data.ball_id}
      className="group rounded-2xl gradient-card border border-border shadow-soft hover:shadow-elegant transition-smooth hover:-translate-y-1 overflow-hidden flex flex-col">

      {/* Colour strip — coded by length */}
      <div className={`h-1.5 w-full bg-gradient-to-r ${lengthColor}`} />

      {/* Title row */}
      <div className="flex items-center justify-between px-4 pt-4 pb-2">
        <div className="flex items-center gap-2.5">
          <div className={`h-9 w-9 rounded-xl bg-gradient-to-br ${lengthColor} flex items-center justify-center shadow-soft flex-shrink-0`}>
            <Sparkles className="h-4 w-4 text-white" />
          </div>
          <div>
            <p className="text-[10px] text-muted-foreground font-medium uppercase tracking-wide">Delivery</p>
            <h3 className="text-base font-bold leading-tight">Ball {data.ball_id ?? "?"}</h3>
          </div>
        </div>
        {typeof data.speed === "number" && (
          <span className={`px-2.5 py-1 rounded-full bg-gradient-to-r ${lengthColor} text-white text-xs font-bold shadow-sm`}>
            {data.speed.toFixed(1)} km/h
          </span>
        )}
      </div>

      {/* Stats grid — the shot spans the row, since it is the model's call */}
      <div className="grid grid-cols-2 gap-2 px-4 pb-2 mt-1 flex-1">
        {stats.map(({ key, label, value, Icon }) => (
          <div key={key}
            className={`rounded-xl bg-secondary/50 border border-border/50 p-2.5 group-hover:bg-secondary transition-smooth ${
              key === "shot" ? "col-span-2" : ""
            }`}>
            <div className="flex items-center gap-1 text-[10px] text-muted-foreground font-medium mb-1">
              <Icon className="h-3 w-3 flex-shrink-0" />
              {label}
            </div>
            <p className="text-sm font-semibold leading-tight break-words">{value}</p>
          </div>
        ))}
      </div>

      {/* Extra angles row (when available) */}
      {extras.length > 0 && (
        <div className="flex gap-2 px-4 pb-4">
          {extras.map(({ label, value, Icon }) => (
            <div key={label}
              className="flex-1 rounded-xl bg-secondary/30 border border-border/30 p-2 flex items-center gap-1.5">
              <Icon className="h-3 w-3 text-muted-foreground flex-shrink-0" />
              <span className="text-[10px] text-muted-foreground">{label}</span>
              <span className="text-xs font-semibold ml-auto">{value}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default BallAnalysisCard;
