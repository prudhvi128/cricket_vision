import { useEffect } from "react";
import { Link, useLocation } from "react-router-dom";
import { Upload, Download, TrendingUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import Navbar from "@/components/Navbar";
import BallAnalysisSection from "@/components/BallAnalysisSection";
import SummaryStats from "@/components/SummaryStats";
import PitchMap from "@/components/PitchMap";

const STORAGE_KEY = "cricktrack_last_result";

const Results = () => {
  const location = useLocation();
  let state = location.state as any;

  if (!state?.result) {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) state = JSON.parse(saved);
    } catch { /* ignore */ }
  }

  useEffect(() => {
    if (state?.result) {
      try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); } catch { /* ignore */ }
    }
  }, [state]);

  if (!state?.result) return (
    <div className="min-h-screen flex items-center justify-center bg-background text-center">
      <div>
        <h2 className="text-2xl font-semibold mb-3">No data found</h2>
        <p className="text-muted-foreground mb-5">Please upload and process a video first.</p>
        <Link to="/upload" className="text-primary underline">Go to Upload</Link>
      </div>
    </div>
  );

  const result         = state.result;
  const processingTime = state.processing_time;
  const videoUrl       = result?.video_url?.replace("127.0.0.1", "localhost");
  const deliveries     = result?.json_data?.deliveries || [];

  const handleExport = () => {
    try {
      const blob = new Blob([JSON.stringify(result?.json_data || {}, null, 2)], { type: "application/json" });
      const url  = URL.createObjectURL(blob);
      const a    = Object.assign(document.createElement("a"), { href: url, download: "cricktrack_analysis.json" });
      a.click(); URL.revokeObjectURL(url);
    } catch { alert("Export failed"); }
  };

  return (
    <div className="min-h-screen bg-background">
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
              Processed in {processingTime}s · {deliveries.length} deliveries detected
            </p>
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
        <SummaryStats deliveries={deliveries} processingTime={processingTime} />

        {/* ── 3. VIDEO PLAYER (full width) ── */}
        <section>
          <h2 className="text-2xl md:text-3xl font-bold tracking-tight mb-4">
            Tracked <span className="text-gradient">Video</span>
          </h2>
          <div className="relative rounded-3xl overflow-hidden border border-border shadow-elegant gradient-card">
            <div className="aspect-video bg-black relative">
              {videoUrl
                ? <video key={videoUrl} controls className="w-full h-full object-contain" src={videoUrl} />
                : <div className="flex items-center justify-center h-full text-white text-sm">No video available</div>}
              <div className="absolute top-4 left-4">
                <span className="px-3 py-1 rounded-full bg-black/60 backdrop-blur text-white text-xs font-semibold border border-white/20">
                  ● TRACKING ON
                </span>
              </div>
            </div>
          </div>
        </section>

        {/* ── 4. PITCH MAP (full width) ── */}
        <PitchMap deliveries={deliveries} />

        {/* ── 5. BALL-BY-BALL ANALYSIS (full width) ── */}
        <BallAnalysisSection deliveries={deliveries} />

      </div>
    </div>
  );
};

export default Results;
