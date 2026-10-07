/**
 * adaptDelivery — the projection the pitch map and the ball cards read.
 *
 * The rule these tests exist to hold: a bounce is plotted only when the record
 * can say where it was on the pitch AND which end of that pitch the batter was
 * at. Frame coordinates are not a substitute, because a position in pixels is
 * not a position on a pitch and can land at the wrong end of it.
 */

import { describe, it, expect } from "vitest";
import { adaptDelivery, type Delivery, type PitchBlock } from "@/lib/api";

const ORIENT_BATTER_AT_BOTTOM: PitchBlock["orientation"] = {
  axis: "v",
  batting_end: 1,
  travel_delta: 0.8,
  calibration_frame: 5,
};

const ORIENT_BATTER_AT_TOP: PitchBlock["orientation"] = {
  axis: "v",
  batting_end: 0,
  travel_delta: -0.8,
  calibration_frame: 5,
};

const ORIENT_CROSSWISE: PitchBlock["orientation"] = {
  axis: "u",
  batting_end: 0,
  travel_delta: -0.5,
  calibration_frame: 5,
};

interface Overrides {
  orientation?: PitchBlock["orientation"];
  pitchPosition?: { x: number; y: number } | null;
  bouncePx?: { x: number; y: number } | null;
  shot?: Delivery["shot"];
}

function delivery(o: Overrides = {}): Delivery {
  const frame = 10;
  const position =
    o.pitchPosition === undefined
      ? null
      : o.pitchPosition && { ...o.pitchPosition, inside_pitch: true };
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
      points: [
        { frame, x: 320, y: 200, source: "detected", confidence: 0.9, pitch_position: position },
      ],
      detected_points: 1,
      predicted_points: 0,
      continuity: "ok",
    },
    bounce: {
      detected: true,
      frame,
      x: o.bouncePx?.x ?? null,
      y: o.bouncePx?.y ?? null,
      ground_x_m: null,
      ground_y_m: null,
    },
    bowling: {
      speed_kmh: 130.4,
      line: "Middle",
      length: "Good Length",
      swing: "",
      release_angle: null,
      bounce_angle: null,
      calibrated: false,
    },
    shot: o.shot ?? { classification: null, confidence: null },
    quality: {
      trajectory_valid: true,
      tracking_confidence: 0.9,
      requires_human_review: false,
      flags: [],
    },
    pitch: {
      state: "calibrated",
      detected: true,
      calibrated: true,
      confidence: 0.9,
      using_previous_calibration: false,
      keypoints: {},
      corners: {},
      center: null,
      homography_available: true,
      orientation: o.orientation ?? null,
    },
  };
}

describe("bounce placement", () => {
  it("plots across and along the pitch with the batter at y = 1", () => {
    const flat = adaptDelivery(
      delivery({
        orientation: ORIENT_BATTER_AT_BOTTOM,
        pitchPosition: { x: 0.4, y: 0.9 },
      }),
      0,
    );
    expect(flat.bounce_x).toBe(0.4);
    expect(flat.bounce_y).toBe(0.9);
  });

  it("turns the point around when the batter is at the other end", () => {
    const flat = adaptDelivery(
      delivery({
        orientation: ORIENT_BATTER_AT_TOP,
        pitchPosition: { x: 0.4, y: 0.9 },
      }),
      0,
    );
    expect(flat.bounce_x).toBe(0.4);
    expect(flat.bounce_y).toBeCloseTo(0.1, 10);
  });

  it("reads across the v axis when the quad lies across the frame", () => {
    const flat = adaptDelivery(
      delivery({
        orientation: ORIENT_CROSSWISE,
        pitchPosition: { x: 0.7, y: 0.25 },
      }),
      0,
    );
    expect(flat.bounce_x).toBe(0.25);
    expect(flat.bounce_y).toBeCloseTo(0.3, 10);
  });

  it("plots nothing when the quad could not be oriented", () => {
    const flat = adaptDelivery(
      delivery({ orientation: null, pitchPosition: { x: 0.4, y: 0.9 } }),
      0,
    );
    expect(flat.bounce_x).toBeUndefined();
    expect(flat.bounce_y).toBeUndefined();
  });

  it("never substitutes frame coordinates for a pitch coordinate", () => {
    const flat = adaptDelivery(
      delivery({
        orientation: ORIENT_BATTER_AT_BOTTOM,
        pitchPosition: null,
        bouncePx: { x: 320, y: 200 },
      }),
      0,
    );
    expect(flat.bounce_x).toBeUndefined();
    expect(flat.bounce_y).toBeUndefined();
  });
});

describe("shot", () => {
  it("surfaces the classification with the confidence it was given", () => {
    const flat = adaptDelivery(
      delivery({ shot: { classification: "Cover", confidence: 0.23131 } }),
      0,
    );
    expect(flat.shot).toBe("Cover");
    expect(flat.shot_confidence).toBeCloseTo(0.23131, 5);
  });

  it("leaves the shot out when the model said nothing", () => {
    const flat = adaptDelivery(
      delivery({ shot: { classification: null, confidence: null } }),
      0,
    );
    expect(flat.shot).toBeUndefined();
    expect(flat.shot_confidence).toBeUndefined();
  });
});
