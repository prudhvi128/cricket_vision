import { useEffect, useMemo, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import {
  Upload, Download, TrendingUp, Loader2, AlertCircle,
  ChevronLeft, ChevronRight,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import Navbar from "@/components/Navbar";
import BallAnalysisSection from "@/components/BallAnalysisSection";
import SummaryStats from "@/components/SummaryStats";
import PitchMap from "@/components/PitchMap";
import PitchOverlay from "@/components/PitchOverlay";
import TrajectoryOverlay from "@/components/TrajectoryOverlay";
import {
  type Delivery,
  type ResultDocument,
  adaptDeliveries,
  cacheResult,
  fetchResult,
  getStoredAnalysisId,
  loadCachedResult,
  mediaUrl,
} from "@/lib/api";

/**
 * Results renders `GET /api/analysis/{CURRENT_ID}/result` and nothing else.
 *
 * THE CURRENT ANALYSIS IS ONE ID
 * ------------------------------
 * The id comes from navigation state when processing just finished, and from
 * localStorage when the page is reloaded. Every piece of state on this page is
 * keyed to that id:
 *
 *   * the cached envelope is only read through `loadCachedResult(analysisId)`,
 *     which returns null unless the copy IS this analysis — so a previous
 *     run's result can never paint here, whether the page is opened directly,
 *     refreshed, or shown while a new analysis is still running;
 *   * a fetched envelope is stored with the id it was requested under and is
 *     rendered only while that id is still the current one;
 *   * with no valid id there is nothing to show, and the page says so instead
 *     of falling back to whatever was cached last.
 */
interface NavState {
  analysis_id?: string;
  processing_time?: number;
}

/** The envelope plus the id it was fetched for — never one without the other. */
interface FetchedResult {
  id: string;
  doc: ResultDocument;
  processingTime?: number;
}

const Results = () => {
  const location = useLocation();
  const navState = (location.state ?? null) as NavState | null;

  // Recomputed on every render, never frozen: a navigation that changes the
  // analysis id changes what this page is allowed to display.
  const analysisId: string | null = navState?.analysis_id ?? getStoredAnalysisId();

  // Paint-from-cache, but only for THIS analysis (null for any other id).
  const cached = useMemo(() => loadCachedResult(analysisId), [analysisId]);

  const [fetched, setFetched] = useState<FetchedResult | null>(null);
  const [loading, setLoading] = useState(Boolean(analysisId));
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!analysisId) {
      setLoading(false);
      setError(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetchResult(analysisId)
      .then((doc) => {
        if (cancelled) return;
        // Belt and braces: a document that says it belongs to another
        // analysis is not this analysis's result and is not rendered.
        if (doc.analysis_id && doc.analysis_id !== analysisId) {
          setError(`The server returned analysis ${doc.analysis_id} instead of ${analysisId}.`);
          return;
        }
        const processingTime = navState?.processing_time ?? cached?.processing_time;
        setFetched({ id: analysisId, doc, processingTime });
        setError(null);
        cacheResult({ ...doc, analysis_id: analysisId, processing_time: processingTime });
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Could not load the result.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [analysisId, navState, cached]);

  /** What this page is allowed to show: this analysis, and only this one. */
  const envelope: ResultDocument | null =
    fetched && fetched.id === analysisId ? fetched.doc : cached;

  const processingTime =
    navState?.processing_time ?? fetched?.processingTime ?? cached?.processing_time;

  /** True while the only thing we have for this id is the local copy. */
  const showingCachedCopy = envelope !== null && envelope !== fetched?.doc;

  const deliveries: Delivery[] | null = envelope?.deliveries ?? null;
  const flat = useMemo(() => adaptDeliveries(deliveries ?? []), [deliveries]);

  /**
   * The clip shown is a SELECTED delivery, navigable with Previous/Next.
   *
   * The initial selection prefers a delivery the pitch model calibrated — the
   * one most likely to have full metrics — then the first with a clip, then
   * the first overall. A delivery list that changes (a new analysis loads)
   * resets the selection to that same preference rather than leaving an index
   * that may now point at nothing.
   */
  const initialIndex = useMemo(() => {
    const list = deliveries ?? [];
    const calibrated = list.findIndex((d) => d?.pitch?.calibrated);
    if (calibrated >= 0) return calibrated;
    const withClip = list.findIndex((d) => Boolean(d?.video?.clip_url));
    return withClip >= 0 ? withClip : 0;
  }, [deliveries]);

  const [selected, setSelected] = useState(initialIndex);
  useEffect(() => {
    setSelected(initialIndex);
  }, [initialIndex]);

  const clamp = (i: number, list: Delivery[]) =>
    Math.max(0, Math.min(i, list.length - 1));

  const deliveryList = deliveries ?? [];
  const featured = deliveryList[selected] ?? deliveryList[initialIndex] ?? null;
  const hasPrev = selected > 0;
  const hasNext = selected < deliveryList.length - 1;

  const videoUrl = mediaUrl(featured?.video?.clip_url);
  const videoW = featured?.video?.width ?? null;
  const videoH = featured?.video?.height ?? null;

  /**
   * Downloads exactly what `/result` serves — the clean envelope. Nothing
   * internal is reachable from this page, let alone exported.
   */
  const handleExport = () => {
    if (!envelope) {
      alert("Nothing to export yet.");
      return;
    }
    try {
      const blob = new Blob([JSON.stringify(envelope, null, 2)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const a = Object.assign(document.createElement("a"), {
        href: url,
        download: "cricktrack_analysis.json",
      });
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      alert(`Export failed: ${err instanceof Error ? err.message : "unknown error"}`);
    }
  };

  // No current analysis: say so. Never fall back to a previous result.
  if (!analysisId) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background text-center px-6">
        <div>
          <AlertCircle className="mx-auto mb-3 h-8 w-8 text-destructive" />
          <h2 className="text-2xl font-semibold mb-3">No data found</h2>
          <p className="text-muted-foreground mb-5">
            No current analysis. Please upload and process a video first.
          </p>
          <Link to="/upload" className="text-primary underline">Go to Upload</Link>
        </div>
      </div>
    );
  }

  if (!deliveries) {
    if (loading) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-background">
          <div className="flex items-center gap-3 text-muted-foreground">
            <Loader2 className="h-5 w-5 animate-spin" /> Loading analysis…
          </div>
        </div>
      );
    }
    return (
      <div className="min-h-screen flex items-center justify-center bg-background text-center px-6">
        <div>
          <AlertCircle className="mx-auto mb-3 h-8 w-8 text-destructive" />
          <h2 className="text-2xl font-semibold mb-3">
            {error ? "Could not load this result" : "No data found"}
          </h2>
          <p className="text-muted-foreground mb-5">
            {error ?? "Please upload and process a video first."}
          </p>
          {/* Which analysis this page asked for — so a failed load names the
              id rather than leaving a previous result to stand in for it. */}
          <p className="text-xs text-muted-foreground font-mono mb-5"
             data-testid="results-analysis-id">
            analysis {analysisId}
          </p>
          <Link to="/upload" className="text-primary underline">Go to Upload</Link>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background" data-analysis-id={analysisId}>
      <Navbar />
      <div className="container py-10 md:py-14 space-y-12">

        {/* ── 1. HEADER ── */}
        <div className="flex flex-wrap items-end justify-between gap-4 animate-fade-in">
          <div className="space-y-2">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 border border-primary/20 text-xs font-semibold text-primary">
              <TrendingUp className="h-3 w-3" /> ANALYSIS COMPLETE
            </div>
            <h1 className="text-4xl md:text-5xl font-bold tracking-tight">
              Match <span className="text-gradient">Analysis</span>
            </h1>
            <p className="text-muted-foreground">
              {processingTime ? `Processed in ${processingTime}s · ` : ""}
              {deliveries.length} deliveries detected
            </p>
            {/* Which analysis this page is showing — the id every value below
                came from, so a wrong result is visible at a glance. */}
            <p className="text-xs text-muted-foreground font-mono" data-testid="results-analysis-id">
              analysis {analysisId}
            </p>
            {error && showingCachedCopy && (
              <p className="text-xs text-amber-500">
                Showing the last saved copy of this analysis — refresh failed: {error}
              </p>
            )}
          </div>
          <div className="flex gap-3">
            <Button variant="outline" className="border-2" onClick={handleExport}>
              <Download className="mr-2 h-4 w-4" /> Export JSON
            </Button>
            <Button asChild className="gradient-primary text-primary-foreground shadow-soft">
              <Link to="/upload"><Upload className="mr-2 h-4 w-4" /> New Video</Link>
            </Button>
          </div>
        </div>

        {/* ── 2. SUMMARY STATS ── */}
        <SummaryStats deliveries={flat} processingTime={processingTime} />

        {/* ── 3. VIDEO PLAYER (full width) ── */}
        <section>
          <div className="flex flex-wrap items-end justify-between gap-4 mb-4">
            <h2 className="text-2xl md:text-3xl font-bold tracking-tight">
              Tracked <span className="text-gradient">Video</span>
            </h2>
            {/* Previous/Next: each swaps the player, overlays and caption for
                the neighbouring delivery WHOLE — same envelope, one index. */}
            {deliveryList.length > 1 && (
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  disabled={!hasPrev}
                  onClick={() => setSelected((i) => clamp(i - 1, deliveryList))}
                  data-testid="prev-delivery"
                >
                  <ChevronLeft className="h-4 w-4" /> Previous
                </Button>
                <span className="text-sm text-muted-foreground font-mono tabular-nums">
                  {selected + 1} / {deliveryList.length}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  disabled={!hasNext}
                  onClick={() => setSelected((i) => clamp(i + 1, deliveryList))}
                  data-testid="next-delivery"
                >
                  Next <ChevronRight className="h-4 w-4" />
                </Button>
              </div>
            )}
          </div>
          <div className="relative rounded-3xl overflow-hidden border border-border shadow-elegant gradient-card">
            <div
              className="bg-black relative w-full"
              style={{
                aspectRatio:
                  videoW && videoH ? `${videoW} / ${videoH}` : undefined,
              }}
            >
              {videoUrl ? (
                <video key={videoUrl} controls className="w-full h-full object-contain" src={videoUrl} />
              ) : (
                <div className="flex items-center justify-center h-full text-white text-sm">
                  No video available
                </div>
              )}
              <div className="absolute top-4 left-4">
                <span className="px-3 py-1 rounded-full bg-black/60 backdrop-blur text-white text-xs font-semibold border border-white/20">
                  ● TRACKING ON
                </span>
              </div>
              <TrajectoryOverlay
                points={featured?.trajectory?.points}
                bounceFrame={featured?.bounce?.frame}
                width={videoW}
                height={videoH}
              />
              <PitchOverlay
                pitch={featured?.pitch}
                width={videoW}
                height={videoH}
              />
            </div>
            {/* The selected delivery's own metrics, read from the same
                envelope the cards below use — one source, two views. */}
            {featured && (
              <div
                className="flex flex-wrap items-center gap-x-4 gap-y-1 px-4 py-3 border-t border-border bg-background/60 backdrop-blur text-sm"
                data-testid="delivery-caption"
              >
                <span className="font-bold">Ball {featured.id}</span>
                <span className="text-muted-foreground">
                  Speed{" "}
                  <span className="font-semibold text-foreground">
                    {typeof featured.bowling?.speed_kmh === "number"
                      ? `${featured.bowling.speed_kmh.toFixed(1)} km/h`
                      : "—"}
                  </span>
                </span>
                <span className="text-muted-foreground">
                  Length{" "}
                  <span className="font-semibold text-foreground">
                    {featured.bowling?.length ?? "—"}
                  </span>
                </span>
                <span className="text-muted-foreground">
                  Line{" "}
                  <span className="font-semibold text-foreground">
                    {featured.bowling?.line ?? "—"}
                  </span>
                </span>
                <span className="text-muted-foreground">
                  Shot{" "}
                  <span className="font-semibold text-foreground">
                    {featured.shot?.classification ?? "—"}
                  </span>
                </span>
                {/* The gate's recorded refusal, when it refused: the reason
                    metrics may be missing is a fact on the record, so it is
                    shown rather than left as unexplained dashes. */}
                {!featured.pitch?.calibrated && featured.pitch?.geometry_reason && (
                  <span className="text-xs text-amber-500 ml-auto">
                    metrics unavailable: {featured.pitch.geometry_reason.replace(/_/g, " ")}
                  </span>
                )}
              </div>
            )}
          </div>
        </section>

        {/* ── 4. PITCH MAP (full width) — same envelope as the cards above */}
        <PitchMap deliveries={flat} />

        {/* ── 5. BALL-BY-BALL ANALYSIS (full width) */}
        <BallAnalysisSection deliveries={flat} />

      </div>
    </div>
  );
};

export default Results;
