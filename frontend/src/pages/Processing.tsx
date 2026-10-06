import { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { Activity, CheckCircle2, Loader2, AlertCircle } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";
import Navbar from "@/components/Navbar";

const STAGES = [
  "Uploading video…",
  "Extracting frames…",
  "Detecting ball position…",
  "Computing trajectory…",
  "Finalizing analysis…",
];

const Processing = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const fileName = (location.state as { fileName?: string })?.fileName ?? "your video";

  const [progress, setProgress]     = useState(0);
  const [stage, setStage]           = useState(0);
  const [errorMsg, setErrorMsg]     = useState<string | null>(null);

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res  = await fetch("http://127.0.0.1:8000/status");
        const data = await res.json();

        if (data.status === "processing") setProgress(data.progress || 10);

        if (data.status === "completed") {
          clearInterval(interval);
          setProgress(100);
          toast.success("Analysis complete!");
          setTimeout(() => navigate("/results", {
            state: { result: data.result, processing_time: data.processing_time },
          }), 800);
        }

        if (data.status === "error") {
          clearInterval(interval);
          setErrorMsg(data.error_message || data.result?.error || "Processing failed.");
          toast.error("Processing failed");
        }
      } catch {
        toast.error("Cannot reach backend — is it running on port 8000?");
      }
    }, 1000);
    return () => clearInterval(interval);
  }, [navigate]);

  useEffect(() => {
    setStage(Math.min(STAGES.length - 1, Math.floor((progress / 100) * STAGES.length)));
  }, [progress]);

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
