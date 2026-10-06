/**
 * PitchMap.tsx — Full broadcast-style 3D-perspective cricket pitch SVG.
 *
 * Design:
 *  - Realistic textured pitch with perspective angle (25-40°)
 *  - All 6 length zones colour-banded
 *  - All 5 line zones
 *  - Accurate stumps, crease lines, popping crease
 *  - Glowing bounce dots positioned from bounce_x / bounce_y
 *  - Hover tooltip with ball stats
 *  - Filter by length zone
 *  - Trajectory line connecting all bounce points in order
 */

import { useState, useMemo } from "react";

interface Delivery {
  ball: number;
  speed: number;
  length: string;
  line: string;
  swing: string;
  release_angle?: number;
  bounce_angle?: number;
  bounce_x?: number;
  bounce_y?: number;
}
interface Props { deliveries: Delivery[]; }

const COLORS: Record<string, string> = {
  Beamer:        "#a855f7",
  Bouncer:       "#ec4899",
  Short:         "#3b82f6",
  "Good Length": "#22c55e",
  Full:          "#f97316",
  Yorker:        "#f43f5e",
  Unknown:       "#94a3b8",
};

const LENGTH_ZONES = [
  { label: "Beamer",       y0: 0.00, y1: 0.30 },
  { label: "Bouncer",      y0: 0.30, y1: 0.45 },
  { label: "Short",        y0: 0.45, y1: 0.58 },
  { label: "Good Length",  y0: 0.58, y1: 0.72 },
  { label: "Full",         y0: 0.72, y1: 0.83 },
  { label: "Yorker",       y0: 0.83, y1: 1.00 },
];

// SVG canvas
const TW = 560, TH = 680;
// Pitch strip — wider at bottom (perspective)
const PITCH_TOP_X   = 200, PITCH_TOP_W   = 160;
const PITCH_BOT_X   = 140, PITCH_BOT_W   = 280;
const PITCH_TOP_Y   = 60,  PITCH_BOT_Y   = 620;
const PH = PITCH_BOT_Y - PITCH_TOP_Y;

/** Map normalised (0-1) bounce coords to SVG perspective coords */
function mapToPitch(bx: number, by: number): [number, number] {
  // Linear interpolation between top and bottom trapezoid edges
  const topX = PITCH_TOP_X + bx * PITCH_TOP_W;
  const botX = PITCH_BOT_X + bx * PITCH_BOT_W;
  const svgX = topX + (botX - topX) * by;
  const svgY = PITCH_TOP_Y + by * PH;
  return [svgX, svgY];
}

const PitchMap = ({ deliveries }: Props) => {
  const [hovered, setHovered] = useState<number | null>(null);
  const [filterLen, setFilterLen] = useState<string>("All");

  const withCoords = useMemo(
    () => deliveries.filter((d) => d.bounce_x !== undefined && d.bounce_y !== undefined),
    [deliveries]
  );

  const filtered = useMemo(
    () => filterLen === "All" ? withCoords : withCoords.filter((d) => d.length === filterLen),
    [withCoords, filterLen]
  );

  // Build perspective polygon points for a zone band
  const zonePoints = (y0: number, y1: number) => {
    const [tlx, tly] = mapToPitch(0, y0);
    const [trx, try_] = mapToPitch(1, y0);
    const [blx, bly] = mapToPitch(0, y1);
    const [brx, bry] = mapToPitch(1, y1);
    return `${tlx},${tly} ${trx},${try_} ${brx},${bry} ${blx},${bly}`;
  };

  // Trapezoid outline points
  const pitchOutline = `
    ${PITCH_TOP_X},${PITCH_TOP_Y}
    ${PITCH_TOP_X + PITCH_TOP_W},${PITCH_TOP_Y}
    ${PITCH_BOT_X + PITCH_BOT_W},${PITCH_BOT_Y}
    ${PITCH_BOT_X},${PITCH_BOT_Y}
  `;

  // Crease line points (horizontal across pitch at given y fraction)
  const creaseLine = (yFrac: number) => {
    const [lx, ly] = mapToPitch(0, yFrac);
    const [rx, ry] = mapToPitch(1, yFrac);
    return { x1: lx, y1: ly, x2: rx, y2: ry };
  };

  // Stump positions at top and bottom crease
  const stumpX = [0.42, 0.50, 0.58];
  const stumpLen = 14;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <h2 className="text-2xl md:text-3xl font-bold tracking-tight">
          Pitch <span className="text-gradient">Map</span>
        </h2>

        {/* Filter buttons */}
        <div className="flex flex-wrap gap-2">
          {["All", ...Object.keys(COLORS).filter(k => k !== "Unknown")].map((len) => (
            <button key={len} onClick={() => setFilterLen(len)}
              className={`px-3 py-1 rounded-full text-xs font-semibold border transition-all ${
                filterLen === len
                  ? "bg-primary text-primary-foreground border-primary"
                  : "border-border text-muted-foreground hover:border-primary/50"
              }`}
              style={filterLen === len ? {} : { borderColor: len !== "All" ? COLORS[len] + "60" : undefined }}>
              {len}
            </button>
          ))}
        </div>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-3">
        {Object.entries(COLORS).filter(([k]) => k !== "Unknown").map(([label, color]) => (
          <div key={label} className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <span className="inline-block h-3 w-3 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />
            {label}
          </div>
        ))}
      </div>

      <div className="w-full rounded-2xl gradient-card border border-border shadow-soft p-6 flex justify-center overflow-x-auto">
        <svg viewBox={`0 0 ${TW} ${TH}`} style={{ width: "100%", maxWidth: `${TW}px`, height: "auto" }}
          aria-label="Cricket pitch map">
          <defs>
            <filter id="dot-glow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="3.5" result="blur" />
              <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
            <filter id="zone-glow">
              <feGaussianBlur stdDeviation="1" result="blur" />
              <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
            {/* Pitch grass gradient */}
            <linearGradient id="pitchGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%"   stopColor="#1a5c2a" />
              <stop offset="50%"  stopColor="#1e7a34" />
              <stop offset="100%" stopColor="#16522a" />
            </linearGradient>
            {/* Pitch strip lighter colour */}
            <linearGradient id="stripGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%"   stopColor="#c8a96e" />
              <stop offset="100%" stopColor="#b8955a" />
            </linearGradient>
            {/* Shadow at pitch edges */}
            <linearGradient id="shadowLeft" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%"   stopColor="#000000" stopOpacity="0.3" />
              <stop offset="100%" stopColor="#000000" stopOpacity="0" />
            </linearGradient>
            <linearGradient id="shadowRight" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%"   stopColor="#000000" stopOpacity="0" />
              <stop offset="100%" stopColor="#000000" stopOpacity="0.3" />
            </linearGradient>
          </defs>

          {/* ── Background ── */}
          <rect x="0" y="0" width={TW} height={TH} fill="#0f1a0f" rx="12" />

          {/* ── Outfield (green) ── */}
          <rect x="0" y="0" width={TW} height={TH} fill="url(#pitchGrad)" rx="12" />

          {/* ── Pitch strip (sand/clay) ── */}
          <polygon points={pitchOutline} fill="url(#stripGrad)" />

          {/* ── Zone colour bands (semi-transparent) ── */}
          {LENGTH_ZONES.map(({ label, y0, y1 }) => (
            <polygon key={label}
              points={zonePoints(y0, y1)}
              fill={COLORS[label] ?? "#888"}
              fillOpacity="0.13"
              stroke={COLORS[label] ?? "#888"}
              strokeOpacity="0.4"
              strokeWidth="0.8"
            />
          ))}

          {/* ── Edge shadows for depth ── */}
          <polygon points={pitchOutline} fill="url(#shadowLeft)"  fillOpacity="0.5" />
          <polygon points={pitchOutline} fill="url(#shadowRight)" fillOpacity="0.5" />

          {/* ── Crease lines ── */}
          {/* Bowling crease top (y=0.09) */}
          {[0.09, 0.13, 0.87, 0.91].map((frac) => {
            const l = creaseLine(frac);
            return <line key={frac} {...l} stroke="white" strokeWidth={frac===0.09||frac===0.91?"1.8":"1"}
              strokeOpacity={frac===0.09||frac===0.91?"0.9":"0.55"}
              strokeDasharray={frac===0.13||frac===0.87?"5 4":""} />;
          })}

          {/* ── Centre line ── */}
          {(() => {
            const [tx, ty] = mapToPitch(0.5, 0.09);
            const [bx, by] = mapToPitch(0.5, 0.91);
            return <line x1={tx} y1={ty} x2={bx} y2={by}
              stroke="white" strokeWidth="0.8" strokeOpacity="0.25" strokeDasharray="4 6" />;
          })()}

          {/* ── Stumps (top bowling end) ── */}
          {stumpX.map((sx) => {
            const [x1, y1] = mapToPitch(sx, 0.09);
            const [x2, y2] = mapToPitch(sx, 0.09);
            return (
              <line key={`ts${sx}`}
                x1={x1} y1={y1 - stumpLen} x2={x2} y2={y2 + 2}
                stroke="#fffde0" strokeWidth="2.5" strokeLinecap="round" />
            );
          })}

          {/* ── Stumps (bottom batting end) ── */}
          {stumpX.map((sx) => {
            const [x1, y1] = mapToPitch(sx, 0.91);
            return (
              <line key={`bs${sx}`}
                x1={x1} y1={y1 - 2} x2={x1} y2={y1 + stumpLen}
                stroke="#fffde0" strokeWidth="2.5" strokeLinecap="round" />
            );
          })}

          {/* ── Bail lines ── */}
          {[0.09, 0.91].map((yf) => {
            const [lx, ly] = mapToPitch(stumpX[0], yf);
            const [rx, ry] = mapToPitch(stumpX[2], yf);
            const sign = yf < 0.5 ? -1 : 1;
            return <line key={`bail${yf}`}
              x1={lx} y1={ly + sign * (yf < 0.5 ? stumpLen : -stumpLen + 2)}
              x2={rx} y2={ry + sign * (yf < 0.5 ? stumpLen : -stumpLen + 2)}
              stroke="#fffde0" strokeWidth="1.8" />;
          })}

          {/* ── Length zone labels (right side) ── */}
          {LENGTH_ZONES.map(({ label, y0, y1 }) => {
            const mid = (y0 + y1) / 2;
            const [lx, ly] = mapToPitch(1, mid);
            return (
              <text key={`lbl${label}`} x={lx + 10} y={ly}
                fontSize="10" fill={COLORS[label]} fontWeight="600"
                dominantBaseline="middle" opacity="0.85">
                {label}
              </text>
            );
          })}

          {/* ── Line zone labels (top) ── */}
          {[
            { label: "Wide Leg",  x: 0.09 },
            { label: "Leg",       x: 0.28 },
            { label: "Middle",    x: 0.50 },
            { label: "Off",       x: 0.72 },
            { label: "Wide Off",  x: 0.91 },
          ].map(({ label, x }) => {
            const [sx, sy] = mapToPitch(x, 0);
            return (
              <text key={label} x={sx} y={sy - 10}
                fontSize="9" fill="#9ca3af" textAnchor="middle" fontWeight="500">
                {label}
              </text>
            );
          })}

          {/* ── Trajectory line connecting bounce points ── */}
          {filtered.length > 1 && (
            <polyline
              points={filtered.map((d) => {
                const [sx, sy] = mapToPitch(d.bounce_x!, d.bounce_y!);
                return `${sx},${sy}`;
              }).join(" ")}
              fill="none"
              stroke="rgba(255,255,255,0.15)"
              strokeWidth="1"
              strokeDasharray="4 4"
            />
          )}

          {/* ── Bounce dots ── */}
          {filtered.map((d, i) => {
            const [sx, sy] = mapToPitch(d.bounce_x!, d.bounce_y!);
            const color    = COLORS[d.length] ?? COLORS.Unknown;
            const isHov    = hovered === i;

            return (
              <g key={i}
                onMouseEnter={() => setHovered(i)}
                onMouseLeave={() => setHovered(null)}
                style={{ cursor: "pointer" }}>

                {/* Outer glow ring */}
                <circle cx={sx} cy={sy} r={isHov ? 16 : 11}
                  fill={color} fillOpacity={isHov ? 0.25 : 0.15}
                  filter="url(#dot-glow)" />

                {/* Main dot */}
                <circle cx={sx} cy={sy} r={isHov ? 8 : 6}
                  fill={color} stroke="white" strokeWidth={isHov ? 1.5 : 1}
                  filter="url(#dot-glow)" />

                {/* Ball number */}
                <text x={sx} y={sy - (isHov ? 14 : 11)}
                  textAnchor="middle" fontSize={isHov ? 10 : 8}
                  fill="white" fontWeight="700">
                  {d.ball}
                </text>

                {/* Hover tooltip */}
                {isHov && (
                  <g>
                    <rect x={sx + 12} y={sy - 52} width={130} height={56}
                      rx="6" fill="rgba(0,0,0,0.88)" stroke={color} strokeWidth="1" />
                    <text x={sx + 20} y={sy - 35} fontSize="11" fill={color} fontWeight="700">
                      Ball {d.ball} — {d.length}
                    </text>
                    <text x={sx + 20} y={sy - 20} fontSize="10" fill="#d1d5db">
                      {d.speed > 0 ? `${d.speed} km/h` : "—"} · {d.line}
                    </text>
                    <text x={sx + 20} y={sy - 6} fontSize="10" fill="#9ca3af">
                      {d.swing || "—"}
                    </text>
                  </g>
                )}
              </g>
            );
          })}

          {/* ── No data message ── */}
          {withCoords.length === 0 && (
            <text x={TW/2} y={TH/2} textAnchor="middle" fontSize="14"
              fill="#9ca3af">
              No bounce data available
            </text>
          )}
        </svg>

        {withCoords.length > 0 && (
          <p className="text-xs text-muted-foreground mt-3 text-center italic w-full">
            Bounce points plotted from model output · hover for ball details
          </p>
        )}
      </div>
    </div>
  );
};

export default PitchMap;
