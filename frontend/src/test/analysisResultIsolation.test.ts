/**
 * One analysis, one result — the storage rules that keep a previous run's
 * envelope from being painted as the current run's result.
 *
 * `loadCachedResult(id)` is the gate: it answers only for the id it is given.
 * A copy under any other id, an absent id, or a shape written by an older build
 * must all come back null so the page falls back to a real fetch (or to
 * "no data"), never to somebody else's numbers.
 */

import { describe, it, expect, beforeEach } from "vitest";
import {
  adaptDelivery,
  cacheResult,
  clearStoredResult,
  getStoredAnalysisId,
  loadCachedResult,
  setStoredAnalysisId,
  type CachedResult,
  type Delivery,
} from "@/lib/api";

function delivery(overrides: Partial<Delivery> = {}): Delivery {
  return {
    id: 1,
    video: {
      clip_url: null,
      start_frame: 0,
      end_frame: 20,
      fps: 25,
      width: 640,
      height: 360,
      duration_seconds: 0.8,
    },
    trajectory: {
      points: [],
      detected_points: 0,
      predicted_points: 0,
      continuity: "ok",
    },
    bounce: {
      detected: false,
      frame: null,
      x: null,
      y: null,
      ground_x_m: null,
      ground_y_m: null,
    },
    bowling: {
      speed_kmh: null,
      line: null,
      length: null,
      swing: null,
      release_angle: null,
      bounce_angle: null,
      calibrated: false,
    },
    shot: { classification: null, confidence: null },
    quality: {
      trajectory_valid: true,
      tracking_confidence: null,
      requires_human_review: false,
      flags: [],
    },
    pitch: {
      state: "no_calibration",
      detected: false,
      calibrated: false,
      confidence: null,
      using_previous_calibration: false,
      keypoints: {},
      corners: {},
      center: null,
      homography_available: false,
      orientation: null,
    },
    ...overrides,
  };
}

function envelope(analysisId: string): CachedResult {
  return {
    analysis_id: analysisId,
    status: "completed",
    pitch: delivery().pitch,
    deliveries: [delivery()],
    processing_time: 12,
  };
}

beforeEach(() => {
  localStorage.clear();
});

describe("cached result is bound to one analysis id", () => {
  it("returns the copy for the analysis it was written under", () => {
    cacheResult(envelope("analysis-B"));
    expect(loadCachedResult("analysis-B")?.analysis_id).toBe("analysis-B");
  });

  it("refuses a copy that belongs to a different analysis", () => {
    cacheResult(envelope("analysis-A"));
    expect(loadCachedResult("analysis-B")).toBeNull();
  });

  it("returns nothing when there is no valid analysis id", () => {
    cacheResult(envelope("analysis-A"));
    expect(loadCachedResult(null)).toBeNull();
    expect(loadCachedResult(undefined)).toBeNull();
    expect(loadCachedResult("")).toBeNull();
  });

  it("rejects a shape written by an older build", () => {
    localStorage.setItem(
      "cricktrack_last_result",
      JSON.stringify({ result: { json_data: { deliveries: [] } } }),
    );
    expect(loadCachedResult("analysis-A")).toBeNull();
  });

  it("drops the copy when a new analysis starts", () => {
    cacheResult(envelope("analysis-A"));
    clearStoredResult();
    expect(loadCachedResult("analysis-A")).toBeNull();
    expect(localStorage.getItem("cricktrack_last_result")).toBeNull();
  });

  it("keeps the stored id and the result as separate entries", () => {
    setStoredAnalysisId("analysis-B");
    cacheResult(envelope("analysis-A"));
    expect(getStoredAnalysisId()).toBe("analysis-B");
    // The id is current, the copy is not — and the copy must not be usable.
    expect(loadCachedResult(getStoredAnalysisId())).toBeNull();
  });
});

describe("adaptDelivery maps, never invents", () => {
  it("leaves an unmeasured length/line out instead of labelling it Unknown", () => {
    const flat = adaptDelivery(delivery(), 0);
    expect(flat.length).toBeUndefined();
    expect(flat.line).toBeUndefined();
    expect(flat.swing).toBeUndefined();
    expect(flat.speed).toBeUndefined();
    expect(flat.shot).toBeUndefined();
  });

  it("passes the backend speed through unchanged — already km/h", () => {
    const flat = adaptDelivery(
      delivery({ bowling: { ...delivery().bowling, speed_kmh: 102.1, length: "Good Length", line: "Middle" } }),
      6,
    );
    expect(flat.speed).toBe(102.1);
    expect(flat.length).toBe("Good Length");
    expect(flat.line).toBe("Middle");
    expect(flat.ball).toBe(7);
  });
});
