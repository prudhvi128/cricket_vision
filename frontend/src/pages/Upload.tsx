import { useCallback, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Upload as UploadIcon, FileVideo, X, ArrowRight, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import Navbar from "@/components/Navbar";

const ACCEPTED = ["video/mp4", "video/avi", "video/quicktime", "video/x-msvideo", "video/webm"];
const MAX_SIZE_MB = 500;

const UploadPage = () => {
  const navigate = useNavigate();
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile]       = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);

  const handleFile = (f: File) => {
    const okType = ACCEPTED.includes(f.type) || /\.(mp4|avi|mov|webm)$/i.test(f.name);
    if (!okType) { toast.error("Invalid file type. Use MP4, AVI, MOV or WEBM."); return; }
    if (f.size / 1024 / 1024 > MAX_SIZE_MB) {
      toast.error(`File too large (max ${MAX_SIZE_MB} MB).`); return;
    }
    setFile(f);
    setPreview(URL.createObjectURL(f));
    toast.success(`${f.name} selected`);
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f) handleFile(f);
  }, []);

  const clear = () => {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null); setPreview(null);
  };

  const submit = async () => {
    if (!file) { toast.error("Please select a video first."); return; }
    setUploading(true);
    try {
      const body = new FormData();
      body.append("file", file);
      const res = await fetch("http://127.0.0.1:8000/upload", { method: "POST", body });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Upload failed");
      }
      toast.success("Upload successful!");
      navigate("/processing", { state: { fileName: file.name } });
    } catch (err: any) {
      toast.error(err.message || "Upload failed. Is the backend running?");
      setUploading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <div className="container py-12 md:py-20 max-w-4xl">
        <div className="text-center mb-10 space-y-3 animate-fade-in">
          <h1 className="text-4xl md:text-5xl font-bold tracking-tight">
            Upload your <span className="text-gradient">match video</span>
          </h1>
          <p className="text-muted-foreground text-lg">
            Drop your file below — we'll handle the rest.
          </p>
        </div>

        <div
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => !file && !uploading && inputRef.current?.click()}
          className={`relative rounded-3xl border-2 border-dashed transition-smooth ${
            dragging ? "border-primary bg-primary/5 scale-[1.01]" : "border-border hover:border-primary/60 bg-card"
          } ${!file && !uploading ? "cursor-pointer" : ""} p-10 md:p-16 shadow-soft animate-scale-in`}
        >
          <input ref={inputRef} type="file"
            accept="video/mp4,video/avi,video/quicktime,video/webm,.mp4,.avi,.mov,.webm"
            className="hidden"
            onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
          />

          {!file ? (
            <div className="flex flex-col items-center text-center space-y-5">
              <div className="flex h-20 w-20 items-center justify-center rounded-2xl gradient-primary shadow-glow animate-float">
                <UploadIcon className="h-10 w-10 text-primary-foreground" />
              </div>
              <div className="space-y-2">
                <p className="text-xl font-semibold">Drag & drop your video here</p>
                <p className="text-muted-foreground">or <span className="text-primary font-medium">browse</span> from your device</p>
              </div>
              <p className="text-xs text-muted-foreground">Supports MP4, AVI, MOV, WEBM · Max {MAX_SIZE_MB} MB</p>
            </div>
          ) : (
            <div className="space-y-6 animate-fade-in">
              <div className="flex items-start justify-between gap-4">
                <div className="flex items-center gap-3 min-w-0">
                  <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl gradient-accent">
                    <FileVideo className="h-6 w-6 text-white" />
                  </div>
                  <div className="min-w-0">
                    <p className="font-semibold truncate">{file.name}</p>
                    <p className="text-sm text-muted-foreground">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                  </div>
                </div>
                {!uploading && (
                  <Button variant="ghost" size="icon" onClick={clear}><X className="h-5 w-5" /></Button>
                )}
              </div>
              {preview && (
                <video src={preview} controls
                  className="w-full rounded-2xl border border-border shadow-soft max-h-96 bg-black" />
              )}
            </div>
          )}
        </div>

        <div className="flex justify-end mt-8">
          <Button size="lg" onClick={submit} disabled={!file || uploading}
            className="gradient-primary text-primary-foreground shadow-elegant hover:shadow-glow transition-smooth h-14 px-8 text-base font-semibold disabled:opacity-50">
            {uploading
              ? <><Loader2 className="mr-2 h-5 w-5 animate-spin" />Uploading…</>
              : <>Analyze Video <ArrowRight className="ml-2 h-5 w-5" /></>}
          </Button>
        </div>
      </div>
    </div>
  );
};

export default UploadPage;
