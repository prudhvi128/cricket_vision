/**
 * api.ts — the single place the browser talks to the backend.
 *
 * Everything else in `src/` goes through these functions, so there is one base
 * URL, one place that knows the six documented paths (docs/API_CONTRACT.md) and
 * one place that decides what a failure looks like to the UI.
 *
 * Two adaptations live here on purpose:
 *
 *   * `adaptDeliveries` projects the `deliveries` array of the `/result`
 *     envelope onto the flat
 *     `{ball, speed, length, line, swing, shot, bounce_x, bounce_y}` shape that
 *     SummaryStats, PitchMap and BallAnalysisSection were written against.
 *     It maps, it never invents: a measurement that is null on the record
 *     comes out absent here, not as zero and not as a label such as
 *     "Unknown" - absent is what the widgets render as "—".
 *   * the analysis id is remembered in localStorage, so a reload of
 *     `/processing` or `/results` continues the analysis it was showing
 *     instead of dead-ending on "no data".
 *
 * ONE ANALYSIS, ONE RESULT
 * ------------------------
 * The cached envelope and the stored id are two separate entries, and they can
 * only be read TOGETHER: `loadCachedResult(analysisId)` refuses a copy whose
 * `analysis_id` is not the id the page is following, so a result from an
 * earlier upload can never paint as if it belonged to the current analysis.
 * `clearStoredResult()` drops the copy the moment a new analysis starts.
 */

export const API_BASE = (
  (import.meta.env.VITE_API_URL as string | undefined) ?? "http://127.0.0.1:8000"
).replace(/\/+$/, "");

const ANALYSIS_ID_KEY = "cricktrack_analysis_id";
const RESULT_KEY = "cricktrack_last_result";

export class ApiError extends Error {
  status?: number;
  constructor(message: string, status?: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export function apiUrl(path: string): string {
  return `${API_BASE}/api${path.startsWith("/") ? path : `/${path}`}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(apiUrl(path), init);
  } catch {
    throw new ApiError("Cannot reach the backend — is it running on port 8000?");
  }
  if (!res.ok) {
    let detail = "";
    try {
      const body = await res.json();
      detail = body?.detail || body?.error || "";
    } catch {
      /* non-JSON error body: fall through to the status line */
    }
    throw new ApiError(detail || `${res.status} ${res.statusText}`, res.status);
  }
  return (await res.json()) as T;
}

/* ── Served types (app.schemas.frontend) ─────────────────────────────────── */

export interface PitchPosition {
  x: number;
  y: number;
  inside_pitch: boolean;
}

/**
 * Which end of the detected quad the batter stood at (`GET /result`).
 *
 * The homography behind `pitch_position` says where the ball was as a fraction
 * of the quad the model found. It says nothing about which end of that quad
 * was the batting end, and a top-down map drawn without it can put a bounce at
 * the bowler's end while the length label beside it says yorker. `null` when no
 * quad could be oriented - which is exactly when the pitch map has to show
 * nothing rather than something possibly backwards.
 */
export interface PitchOrientation {
  axis: "u" | "v" | string;
  batting_end: number;
  travel_delta: number | null;
  calibration_frame: number | null;
}

export interface PitchBlock {
  state: "calibrated" | "temporary_loss" | "no_calibration" | "calibration_expired" | string;
  detected: boolean;
  calibrated: boolean;
  confidence: number | null;
  using_previous_calibration: boolean;
  keypoints: Record<string, number[]>;
  corners: Record<string, number[]>;
  center: number[] | null;
  homography_available: boolean;
  orientation: PitchOrientation | null;
}

export interface TrajectoryPoint {
  frame: number;
  x: number;
  y: number;
  source: string;
  confidence: number | null;
  pitch_position: PitchPosition | null;
}

export interface Delivery {
  id: number;
  video: {
    clip_url: string | null;
    start_frame: number | null;
    end_frame: number | null;
    fps: number | null;
    width: number | null;
    height: number | null;
    duration_seconds: number | null;
  };
  trajectory: {
    points: TrajectoryPoint[];
    detected_points: number;
    predicted_points: number;
    continuity: string;
  };
  bounce: {
    detected: boolean;
    frame: number | null;
    x: number | null;
    y: number | null;
    ground_x_m: number | null;
    ground_y_m: number | null;
  };
  bowling: {
    speed_kmh: number | null;
    line: string | null;
    length: string | null;
    swing: string | null;
    release_angle: number | null;
    bounce_angle: number | null;
    calibrated: boolean;
  };
  shot: { classification: string | null; confidence: number | null };
  quality: {
    trajectory_valid: boolean;
    tracking_confidence: number | null;
    requires_human_review: boolean;
    flags: string[];
  };
  pitch: PitchBlock;
}

/** `GET /result` — one envelope, not a bare array. */
export interface ResultDocument {
  analysis_id: string;
  status: string;
  pitch: PitchBlock;
  deliveries: Delivery[];
}

export interface AnalysisStatus {
  analysis_id: string;
  status: "queued" | "running" | "completed" | "failed" | string;
  stage?: string | null;
  substage?: string | null;
  progress?: number | null;
  detail?: string | null;
  error?: string | null;
  frames_done?: number | null;
  frames_total?: number | null;
  source_name?: string | null;
}

/* ── Calls ───────────────────────────────────────────────────────────────── */

/**
 * A URL the API returned (`/api/analysis/...`) made absolute.
 *
 * The record holds root-relative URLs because the API answers on its own
 * origin, which is NOT the origin the page is served from in development —
 * so a bare `/api/...` would be requested from the Vite dev server and 404.
 */
export function mediaUrl(url?: string | null): string | null {
  if (!url) return null;
  if (/^https?:\/\//i.test(url)) return url;
  if (url.startsWith("/")) return `${API_BASE}${url}`;
  return url;
}

/**
 * `POST /api/analyze` — accepts the video and returns an analysis id.
 *
 * The documented entry point (docs/API_CONTRACT.md); `POST /api/upload`
 * with a `file=` field is the same handler under the other spelling.
 */
export async function uploadVideo(file: File): Promise<{ analysis_id: string }> {
  const body = new FormData();
  body.append("video", file);
  return request<{ analysis_id: string }>("/analyze", {
    method: "POST",
    body,
  });
}

/** `GET /api/analysis/{id}/status` — queued | running | completed | failed. */
export function fetchStatus(analysisId: string): Promise<AnalysisStatus> {
  return request<AnalysisStatus>(`/analysis/${encodeURIComponent(analysisId)}/status`);
}

/** `GET /api/analysis/{id}/result` — the envelope the UI renders and exports. */
export function fetchResult(analysisId: string): Promise<ResultDocument> {
  return request<ResultDocument>(`/analysis/${encodeURIComponent(analysisId)}/result`);
}

/* ── The analysis id ─────────────────────────────────────────────────────── */

export function getStoredAnalysisId(): string | null {
  try {
    return localStorage.getItem(ANALYSIS_ID_KEY);
  } catch {
    return null;
  }
}

export function setStoredAnalysisId(id: string): void {
  try {
    localStorage.setItem(ANALYSIS_ID_KEY, id);
  } catch {
    /* private mode: the id still travels in navigation state */
  }
}

/* ── Result cache (survives a reload) ────────────────────────────────────── */

/**
 * The `/result` envelope as it was last fetched, plus how long the run took.
 * Cached whole so a reload paints without a round trip AND can re-export the
 * same document the server sent.
 *
 * A cache entry is only ever valid for the analysis id it was written under.
 */
export interface CachedResult extends ResultDocument {
  processing_time?: number;
}

export function cacheResult(value: CachedResult): void {
  try {
    localStorage.setItem(RESULT_KEY, JSON.stringify(value));
  } catch {
    /* a large result is not worth crashing the page over */
  }
}

/**
 * Drop the cached envelope. Called when a NEW analysis starts, so the previous
 * analysis cannot be painted while — or after — the new one runs.
 */
export function clearStoredResult(): void {
  try {
    localStorage.removeItem(RESULT_KEY);
  } catch {
    /* private mode: there is nothing to clear anyway */
  }
}

/**
 * The cached envelope for `analysisId`, or null.
 *
 * `analysisId` is required and the copy is checked against it: with no valid
 * id there is nothing the page is allowed to show, and an envelope from a
 * different analysis must not be shown in this one's place. So this returns
 * null when there is no cache, when the copy was written by an older build
 * with a different shape, or when it belongs to another analysis.
 */
export function loadCachedResult(analysisId: string | null | undefined): CachedResult | null {
  if (!analysisId) return null;
  try {
    const raw = localStorage.getItem(RESULT_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed?.deliveries)) return null;
    if (typeof parsed?.analysis_id !== "string") return null;
    // The whole point: another analysis's document is not this analysis's
    // result, however recent it is.
    if (parsed.analysis_id !== analysisId) return null;
    // Rejects a copy written before this envelope existed: it has no pitch
    // block, so exporting it would send a document that does not match the
    // schema the server serves. The fetch on mount replaces it anyway.
    if (typeof parsed?.pitch !== "object" || !parsed.pitch) return null;
    return parsed as CachedResult;
  } catch {
    return null;
  }
}

/* ── The flat projection the existing widgets consume ────────────────────── */

export interface FlatDelivery {
  ball: number;
  /** km/h as the backend serves it, or absent when there is no speed. */
  speed?: number;
  /** Absent when the record has no length — the widgets render "—". */
  length?: string;
  /** Absent when the record has no line — the widgets render "—". */
  line?: string;
  swing?: string;
  /** Shot classification from the model, e.g. "Cover" — absent when unknown. */
  shot?: string;
  /** The model's own confidence for `shot`, 0..1. */
  shot_confidence?: number;
  release_angle?: number;
  bounce_angle?: number;
  /**
   * Where the ball bounced, in map coordinates: x across the pitch (0..1) and
   * y measured from the NON-batting end, so the batting end is always y = 1.
   */
  bounce_x?: number;
  bounce_y?: number;
}

/**
 * Where the ball bounced, as the pitch map wants it.
 *
 * Two conditions, both required, and deliberately no third:
 *
 *   * the trajectory point at the bounce frame must carry a `pitch_position`
 *     (a coordinate produced by the pitch model's homography), and
 *   * the delivery must carry an `orientation`, which says which end of that
 *     quad the batter was at.
 *
 * The frame-coordinate fallback is gone on purpose. A position in frame pixels
 * is not a position on a pitch: plotting one would invent exactly the
 * calibration this map exists to show, and could put a bounce at the wrong end
 * of the strip. A delivery that cannot place its bounce honestly gets no dot.
 *
 * The returned axes are already oriented (batting end at y = 1), which is the
 * same transform `map_homography` performs server-side.
 */
function bouncePosition(delivery: Delivery): { x?: number; y?: number } {
  const { bounce, trajectory, pitch } = delivery;
  const orientation = pitch?.orientation;
  if (!orientation) return {};
  const axis = orientation.axis;
  if (axis !== "u" && axis !== "v") return {};

  const points = trajectory?.points ?? [];
  const atBounce =
    bounce?.frame != null ? points.find((p) => p.frame === bounce.frame) : undefined;
  const position = atBounce?.pitch_position;
  if (!position || !Number.isFinite(position.x) || !Number.isFinite(position.y)) {
    return {};
  }

  const u = position.x;
  const v = position.y;
  const towardsBattingEnd = orientation.batting_end === 1;
  if (axis === "v") {
    // Along-pitch axis is v: across is u, and the batter sits at v = 0 or 1.
    return { x: u, y: towardsBattingEnd ? v : 1 - v };
  }
  // Along-pitch axis is u: across is v, batter at u = 0 or 1.
  return { x: v, y: towardsBattingEnd ? u : 1 - u };
}

export function adaptDelivery(delivery: Delivery, index: number): FlatDelivery {
  const bowling = delivery.bowling ?? ({} as Delivery["bowling"]);
  const shot = delivery.shot;
  const { x, y } = bouncePosition(delivery);
  return {
    ball: index + 1,
    // The backend serves these already in km/h (`speed_kmh`); the unit is
    // never touched here, and a null comes out as "absent", never as a
    // placeholder label — "Unknown" would read as an answer the run gave.
    speed: bowling.speed_kmh ?? undefined,
    length: bowling.length ?? undefined,
    line: bowling.line ?? undefined,
    swing: bowling.swing ?? undefined,
    shot: shot?.classification ?? undefined,
    shot_confidence: shot?.confidence ?? undefined,
    release_angle: bowling.release_angle ?? undefined,
    bounce_angle: bowling.bounce_angle ?? undefined,
    bounce_x: x,
    bounce_y: y,
  };
}

export function adaptDeliveries(deliveries: Delivery[]): FlatDelivery[] {
  return deliveries.map(adaptDelivery);
}
