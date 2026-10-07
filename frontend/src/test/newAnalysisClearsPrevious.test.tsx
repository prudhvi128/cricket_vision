/**
 * Starting a NEW analysis must wipe the previous one's state: the old id goes,
 * the old envelope goes, and nothing from the earlier run is left for
 * /processing or /results to pick up while the new analysis runs.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import UploadPage from "@/pages/Upload";
import { ThemeProvider } from "@/components/ThemeProvider";
import {
  cacheResult,
  getStoredAnalysisId,
  loadCachedResult,
  setStoredAnalysisId,
  type CachedResult,
} from "@/lib/api";

const PREVIOUS = "analysis-A-previous";
const NEW = "analysis-B-new";

function previousEnvelope(): CachedResult {
  return {
    analysis_id: PREVIOUS,
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
    deliveries: [],
    processing_time: 42,
  };
}

const fetchMock = vi.fn();

beforeEach(() => {
  localStorage.clear();
  fetchMock.mockReset();
  fetchMock.mockResolvedValue({
    ok: true,
    status: 202,
    statusText: "Accepted",
    json: async () => ({ analysis_id: NEW, status: "queued" }),
  } as unknown as Response);
  vi.stubGlobal("fetch", fetchMock);
  // jsdom has no object URLs; the preview <video> needs one.
  window.URL.createObjectURL = vi.fn(() => "blob:preview");
  window.URL.revokeObjectURL = vi.fn();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("new analysis", () => {
  it("stores the new id and clears the previous result", async () => {
    setStoredAnalysisId(PREVIOUS);
    cacheResult(previousEnvelope());

    const { container } = render(
      <MemoryRouter initialEntries={["/upload"]}>
        <ThemeProvider>
          <UploadPage />
        </ThemeProvider>
      </MemoryRouter>,
    );

    const input = container.querySelector('input[type="file"]') as HTMLInputElement;
    const file = new File([new Uint8Array([1, 2, 3])], "match.mp4", { type: "video/mp4" });
    Object.defineProperty(input, "files", { value: [file], configurable: true });
    fireEvent.change(input);

    fireEvent.click(await screen.findByRole("button", { name: /Analyze Video/ }));

    await waitFor(() => expect(getStoredAnalysisId()).toBe(NEW));
    // The previous analysis's envelope is gone the moment this one starts.
    expect(localStorage.getItem("cricktrack_last_result")).toBeNull();
    expect(loadCachedResult(PREVIOUS)).toBeNull();
    expect(loadCachedResult(NEW)).toBeNull();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(String(fetchMock.mock.calls[0][0])).toContain("/api/analyze");
  });
});
