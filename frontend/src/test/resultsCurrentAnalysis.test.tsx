/**
 * /results must show the CURRENT analysis and nothing else.
 *
 * The scenarios below are the ones the bug report called out: a previous
 * analysis sitting in localStorage, a page opened with no valid analysis id,
 * and a fetch that fails while that stale copy is still there. In every one of
 * them the previous analysis's numbers must stay off the screen.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import Results from "@/pages/Results";
import { ThemeProvider } from "@/components/ThemeProvider";
import {
  cacheResult,
  clearStoredResult,
  setStoredAnalysisId,
  type CachedResult,
} from "@/lib/api";

const CURRENT = "analysis-B-current";
const PREVIOUS = "analysis-A-previous";

interface Spec {
  /** km/h per delivery; null = the run measured no speed. */
  speeds: (number | null)[];
  lengths?: (string | null)[];
  lines?: (string | null)[];
}

function envelope(analysisId: string, spec: Spec): CachedResult {
  const { speeds, lengths = [], lines = [] } = spec;
  return {
    analysis_id: analysisId,
    status: "completed",
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
    deliveries: speeds.map((speed, i) => ({
      id: i + 1,
      video: {
        clip_url: null,
        start_frame: null,
        end_frame: null,
        fps: null,
        width: null,
        height: null,
        duration_seconds: null,
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
        speed_kmh: speed,
        line: lines[i] ?? null,
        length: lengths[i] ?? null,
        swing: null,
        release_angle: null,
        bounce_angle: null,
        calibrated: false,
      },
      shot: { classification: "Cover", confidence: 0.5 },
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
    })),
    processing_time: 12,
  };
}

const fetchMock = vi.fn();

function respondWith(doc: unknown) {
  fetchMock.mockResolvedValue({
    ok: true,
    status: 200,
    statusText: "OK",
    json: async () => doc,
  } as unknown as Response);
}

function respondWithFailure(status: number, detail: string) {
  fetchMock.mockResolvedValue({
    ok: false,
    status,
    statusText: "Not Found",
    json: async () => ({ detail }),
  } as unknown as Response);
}

function renderResults(navState?: { analysis_id?: string; processing_time?: number }) {
  return render(
    <ThemeProvider>
      <MemoryRouter initialEntries={[{ pathname: "/results", state: navState ?? null }]}>
        <Results />
      </MemoryRouter>
    </ThemeProvider>,
  );
}

beforeEach(() => {
  localStorage.clear();
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("current analysis only", () => {
  it("never paints a previous analysis while the current one cannot be fetched", async () => {
    // A previous run's envelope is the only thing in storage, and the fetch
    // for the CURRENT id fails (404 / still running / backend down).
    cacheResult(envelope(PREVIOUS, { speeds: [145.1], lengths: ["Short"], lines: ["Middle"] }));
    setStoredAnalysisId(CURRENT);
    respondWithFailure(404, "No result for analysis");

    renderResults({ analysis_id: CURRENT });

    expect(await screen.findByText(/Could not load this result/)).toBeInTheDocument();
    expect(screen.queryByText(/145\.1/)).toBeNull();
    expect(screen.getByTestId("results-analysis-id")).toHaveTextContent(CURRENT);
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining(`/api/analysis/${CURRENT}/result`),
      undefined,
    );
  });

  it("shows no data — not a stale result — when there is no valid analysis id", async () => {
    cacheResult(envelope(PREVIOUS, { speeds: [145.1], lengths: ["Short"] }));
    respondWithFailure(404, "No result");

    renderResults();

    expect(await screen.findByText("No data found")).toBeInTheDocument();
    expect(screen.queryByText(/145\.1/)).toBeNull();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("renders exactly the envelope the current analysis id returns", async () => {
    cacheResult(envelope(PREVIOUS, { speeds: [145.1], lengths: ["Short"] }));
    setStoredAnalysisId(CURRENT);
    respondWith(
      envelope(CURRENT, {
        speeds: [84.8, 102.1, 82.3],
        lengths: ["Good Length", "Good Length", null],
        lines: ["Middle", "Middle", null],
      }),
    );

    renderResults({ analysis_id: CURRENT });

    // Summary is computed from THIS envelope: (84.8 + 102.1 + 82.3) / 3.
    expect(await screen.findByTestId("summary-avg-speed")).toHaveTextContent("89.7 km/h");
    expect(screen.getByTestId("summary-total-deliveries")).toHaveTextContent("3");
    expect(screen.getByTestId("summary-top-length")).toHaveTextContent("Good Length");
    expect(screen.getByTestId("results-analysis-id")).toHaveTextContent(CURRENT);
    // The previous analysis's speed is nowhere on the page.
    expect(screen.queryByText(/145\.1/)).toBeNull();
    // Cards carry the backend's own km/h values, unconverted.
    expect(screen.getAllByText("102.1 km/h").length).toBeGreaterThan(0);
    expect(screen.getAllByText("84.8 km/h").length).toBeGreaterThan(0);
    expect(screen.getAllByText("82.3 km/h").length).toBeGreaterThan(0);
  });

  it("rejects a document the server says belongs to another analysis", async () => {
    setStoredAnalysisId(CURRENT);
    respondWith(envelope(PREVIOUS, { speeds: [145.1] }));

    renderResults({ analysis_id: CURRENT });

    expect(await screen.findByText(/Could not load this result/)).toBeInTheDocument();
    expect(screen.queryByText(/145\.1/)).toBeNull();
  });

  it("drops the stale copy as soon as the current analysis's result arrives", async () => {
    cacheResult(envelope(PREVIOUS, { speeds: [145.1] }));
    setStoredAnalysisId(CURRENT);
    respondWith(envelope(CURRENT, { speeds: [90.5] }));

    renderResults({ analysis_id: CURRENT });

    expect(await screen.findByTestId("summary-avg-speed")).toHaveTextContent("90.5 km/h");
    expect(screen.queryByText(/145\.1/)).toBeNull();
    // The storage now holds the current analysis, so a refresh paints the same
    // thing rather than the previous run.
    const stored = JSON.parse(localStorage.getItem("cricktrack_last_result") ?? "{}");
    expect(stored.analysis_id).toBe(CURRENT);
  });

  it("renders unmeasured measurements as —, never as Unknown", async () => {
    clearStoredResult();
    setStoredAnalysisId(CURRENT);
    respondWith(
      envelope(CURRENT, {
        speeds: [null, null],
        lengths: [null, null],
        lines: [null, null],
      }),
    );

    const { container } = renderResults({ analysis_id: CURRENT });

    expect(await screen.findByTestId("summary-total-deliveries")).toHaveTextContent("2");
    // No fabricated label anywhere on the page.
    expect(screen.queryByText("Unknown")).toBeNull();
    // Summary: nothing was measured, so the summary says so.
    expect(screen.getByTestId("summary-avg-speed")).toHaveTextContent("—");
    expect(screen.getByTestId("summary-top-length")).toHaveTextContent("—");
    // The ball cards render the dash too.
    expect(container.querySelectorAll("[data-ball-id]").length).toBe(2);
    expect(screen.getAllByText("—").length).toBeGreaterThan(0);
  });
});
