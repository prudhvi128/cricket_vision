import { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { Activity, CheckCircle2, Loader2, AlertCircle } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";
import Navbar from "@/components/Navbar";
import {
  AnalysisStatus,
  cacheResult,
  clearStoredResult,
  fetchResult,
  fetchStatus,
  getStoredAnalysisId,
  loadCachedResult,
} from "@/lib/api";

const STAGES = [
  "Uploading video…",
  "Extracting frames…",
  "Detecting ball position…",
  "Computing trajectory…",
  "Finalizing analysis…",
];

/**
 * Backend stage → the step list above. The pipeline reports seven stages
 * (docs/API_CONTRACT.md); these five are what the UI shows, so the tracking
 * pass is split by its substage rather than collapsed into one long step.
 */
function stageIndex(status: AnalysisStatus, progress: number): number {
  switch (status.stage) {
    case "uploading":
      return 0;
    case "tracking":
      if (status.substage === "segmentation") return 1;
      if (status.substage === "clip_writing") return 3;
      return 2;
    case "validation":
      return 3;
    case "shot_classification":
    case "overlay":
    case "persisting":
    case "completed":
      return 4;
    default:
      return Math.min(STAGES.length - 1, Math.floor((progress / 100) * STAGES.length));
  }
}

const Processing = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const navState = location.state as
    | { fileName?: string; analysis_id?: string }
    | null;
  const fileName = navState?.fileName ?? "your video";
  const [analysisId] = useState<string | null>(
    navState?.analysis_id ?? getStoredAnalysisId()
  );

  const [progress, setProgress]     = useState(0);
  const [stage, setStage]           = useState(0);
  const [detail, setDetail]         = useState<string>("");
  const [errorMsg, setErrorMsg]     = useState<string | null>(null);

  useEffect(() => {
    if (!analysisId) {
      setErrorMsg("No analysis to follow. Upload a video first.");
      return;
    }

    // This page follows exactly one analysis. A cached envelope that belongs
    // to a different id is a previous run's result and is dropped, so opening
    // /results mid-run can never paint it as this run's output.
    if (!loadCachedResult(analysisId)) clearStoredResult();

    let cancelled = false;
    let failures = 0;
    const started = Date.now();
    let timer: ReturnType<typeof setInterval> | null = null;

    const fail = (message: string) => {
      if (cancelled) return;
      if (timer) clearInterval(timer);
      setErrorMsg(message);
    };

    const poll = async () => {
      if (cancelled) return;
      let status: AnalysisStatus;
      try {
        status = await fetchStatus(analysisId);
        failures = 0;
      } catch (err) {
        const message = err instanceof Error ? err.message : "Cannot reach the backend.";
        failures += 1;
        if (failures === 1) toast.error(message);
        // A transient blip must not kill the page; three in a row is an outage.
        if (failures >= 3) fail(message);
        return;
      }
      if (cancelled) return;

      const value = typeof status.progress === "number" ? status.progress : 0;
      const clamped = Math.max(0, Math.min(100, value));
      setProgress(clamped);
      setStage(stageIndex(status, clamped));
      setDetail(status.detail || "");

      if (status.status === "completed") {
        if (timer) clearInterval(timer);
        setProgress(100);
        const processing_time = Math.max(1, Math.round((Date.now() - started) / 1000));
        const doc = await fetchResult(analysisId).catch(() => null);
        // Only this analysis's envelope is worth remembering; a document that
        // identifies as a different analysis is not cached under this id.
        if (doc && (!doc.analysis_id || doc.analysis_id === analysisId)) {
          cacheResult({ ...doc, analysis_id: analysisId, processing_time });
        }
        if (cancelled) return;
        toast.success("Analysis complete!");
        navigate("/results", { state: { analysis_id: analysisId, processing_time } });
        return;
      }

      if (status.status === "failed") {
        fail(status.error || "Processing failed.");
        toast.error("Processing failed");
      }
    };

    poll();
    timer = setInterval(poll, 1000);
    return () => {
      cancelled = true;
      if (timer) clearInterval(timer);
    };
  }, [analysisId, navigate]);

  if (errorMsg) return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <div className="container py-12 md:py-20 max-w-2xl">
        <div className="rounded-3xl gradient-card border border-destructive/40 p-8 md:p-12 shadow-elegant animate-scale-in">
          <div className="flex flex-col items-center text-center space-y-6">
            <div className="flex h-20 w-20 items-center justify-center rounded-2xl bg-destructive/10 border border-destructive/30">
              <AlertCircle className="h-10 w-10 text-destructive" />
            </div>
            <div className="space-y-2">
              <h1 className="text-3xl font-bold tracking-tight text-destructive">Processing Failed</h1>
              <p className="text-muted-foreground max-w-md">{errorMsg}</p>
            </div>
            <Button asChild className="gradient-primary text-primary-foreground">
              <Link to="/upload">Try Again</Link>
            </Button>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <div className="container py-12 md:py-20 max-w-2xl">
        <div className="rounded-3xl gradient-card border border-border p-8 md:p-12 shadow-elegant animate-scale-in">
          <div className="flex flex-col items-center text-center space-y-6">

            <div className="relative">
              <div className="absolute inset-0 gradient-primary rounded-2xl blur-2xl opacity-60 animate-pulse-glow" />
              <div className="relative flex h-20 w-20 items-center justify-center rounded-2xl gradient-primary shadow-glow">
                <Activity className="h-10 w-10 text-primary-foreground animate-pulse" />
              </div>
            </div>

            <div className="space-y-2">
              <h1 className="text-3xl md:text-4xl font-bold tracking-tight">Processing your video</h1>
              <p className="text-muted-foreground truncate max-w-md">{fileName}</p>
            </div>

            <div className="w-full space-y-3">
              <div className="flex justify-between text-sm font-medium">
                <span className="text-muted-foreground">{STAGES[stage]}</span>
                <span className="text-gradient font-bold">{Math.floor(progress)}%</span>
              </div>
              <div className="h-3 w-full overflow-hidden rounded-full bg-secondary">
                <div className="h-full gradient-accent transition-all duration-300 ease-out"
                  style={{ width: `${progress}%` }} />
              </div>
              {detail && (
                <p className="text-xs text-muted-foreground truncate text-center">{detail}</p>
              )}
            </div>

            <div className="w-full space-y-2 pt-4">
              {STAGES.map((s, i) => (
                <div key={s} className={`flex items-center gap-3 px-4 py-3 rounded-xl ${
                  i < stage  ? "bg-secondary/50 text-foreground"
                  : i === stage ? "bg-primary/10 text-foreground border border-primary/30"
                  : "text-muted-foreground"}`}>
                  {i < stage    ? <CheckCircle2 className="h-5 w-5 text-primary shrink-0" />
                   : i === stage ? <Loader2 className="h-5 w-5 text-primary animate-spin shrink-0" />
                   : <div className="h-5 w-5 rounded-full border-2 border-border shrink-0" />}
                  <span className="text-sm font-medium">{s}</span>
                </div>
              ))}
            </div>

          </div>
        </div>
      </div>
    </div>
  );
};

export default Processing;
