/**
 * TrajectoryOverlay.tsx — the tracked ball path, drawn over the delivery clip.
 *
 * The points come from `Delivery.trajectory.points`, which are in SOURCE pixel
 * coordinates — the same system `PitchOverlay` draws its quad in, and the same
 * viewBox the video stretches with. Nothing is interpolated here: the line is
 * the detections, in time order, and the predicted tail (after the tracker
 * coasted through a loss) is dashed so it never reads as measured.
 *
 * The bounce marker is the trajectory point AT the bounce frame — the one the
 * backend's `bounce` block names — not a recomputed lowest point: the record
 * already decided where the bounce was, and a second answer here would
 * disagree with the pitch map beside it.
 */

import type { TrajectoryPoint } from "@/lib/api";

interface Props {
  points?: TrajectoryPoint[] | null;
  /** The bounce frame the backend recorded, or null. */
  bounceFrame?: number | null;
  /** Source frame width — the coordinate system the points are in. */
  width?: number | null;
  /** Source frame height. */
  height?: number | null;
}

const DETECTED = "#38bdf8";
const PREDICTED = "#94a3b8";
const BOUNCE = "#f472b6";

const TrajectoryOverlay = ({ points, bounceFrame, width, height }: Props) => {
  if (!width || !height || !points || points.length === 0) return null;

  const unit = Math.max(2, Math.min(width, height) * 0.006);
  const valid = points.filter(
    (p) => Number.isFinite(p.x) && Number.isFinite(p.y)
  );
  if (valid.length === 0) return null;

  // One polyline for the measured part (up to the first predicted point) and
  // one for the coasted tail, so the dashed segment starts exactly where the
  // backend says detection stopped.
  const firstPredicted = valid.findIndex((p) => p.source === "predicted");
  const measured = firstPredicted === -1 ? valid : valid.slice(0, Math.max(2, firstPredicted));
  const tail = firstPredicted === -1 ? [] : valid.slice(Math.max(0, firstPredicted - 1));

  const bouncePoint =
    bounceFrame != null ? valid.find((p) => p.frame === bounceFrame) : undefined;

  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden">
      <svg
        className="absolute inset-0 h-full w-full"
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        aria-hidden="true"
      >
        {measured.length >= 2 && (
          <polyline
            points={measured.map((p) => `${p.x},${p.y}`).join(" ")}
            fill="none"
            stroke={DETECTED}
            strokeWidth={Math.max(2, unit * 0.8)}
            strokeOpacity={0.9}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        )}
        {tail.length >= 2 && (
          <polyline
            points={tail.map((p) => `${p.x},${p.y}`).join(" ")}
            fill="none"
            stroke={PREDICTED}
            strokeWidth={Math.max(2, unit * 0.7)}
            strokeOpacity={0.8}
            strokeDasharray={`${unit * 3} ${unit * 2.5}`}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        )}
        {valid.map((p, i) => (
          <circle
            key={`${p.frame}-${i}`}
            cx={p.x}
            cy={p.y}
            r={unit * (p.source === "predicted" ? 0.9 : 1.2)}
            fill={p.source === "predicted" ? PREDICTED : DETECTED}
            fillOpacity={0.9}
            stroke="#0b1220"
            strokeWidth={Math.max(1, unit * 0.35)}
          />
        ))}
        {bouncePoint && (
          <g>
            <circle
              cx={bouncePoint.x}
              cy={bouncePoint.y}
              r={unit * 2.2}
              fill="none"
              stroke={BOUNCE}
              strokeWidth={Math.max(2, unit * 0.8)}
              strokeOpacity={0.95}
            />
            <circle
              cx={bouncePoint.x}
              cy={bouncePoint.y}
              r={unit * 1.1}
              fill={BOUNCE}
              fillOpacity={0.95}
            />
          </g>
        )}
      </svg>
    </div>
  );
};

export default TrajectoryOverlay;
