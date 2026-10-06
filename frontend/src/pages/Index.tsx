import { Link } from "react-router-dom";
import { ArrowRight, Upload, Zap, Target, BarChart3, Play } from "lucide-react";
import { Button } from "@/components/ui/button";
import Navbar from "@/components/Navbar";
import heroImg from "@/assets/hero-cricket.jpg";

const features = [
  { icon: Zap,      title: "Real-time Tracking",    desc: "AI-powered YOLO11 ball detection with Kalman filter smoothing across every frame." },
  { icon: Target,   title: "Trajectory Analysis",   desc: "Visualize swing, spin, and bounce points with glowing IPL-style overlays." },
  { icon: BarChart3,title: "Performance Insights",  desc: "Detailed metrics on speed, line, length and swing for every delivery." },
];

const Index = () => (
  <div className="min-h-screen bg-background">
    <Navbar />

    {/* Hero */}
    <section className="relative overflow-hidden">
      <div className="absolute inset-0 gradient-hero opacity-60" />
      <div className="absolute top-20 -left-40 h-96 w-96 rounded-full bg-primary/20 blur-3xl animate-float" />
      <div className="absolute bottom-0 -right-40 h-96 w-96 rounded-full bg-accent/20 blur-3xl animate-float" style={{ animationDelay: "2s" }} />

      <div className="container relative grid lg:grid-cols-2 gap-12 items-center py-20 lg:py-28">
        <div className="space-y-8 animate-fade-in-up">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-secondary/80 backdrop-blur-sm border border-border text-sm font-medium">
            <span className="h-2 w-2 rounded-full bg-primary animate-pulse" />
            AI-Powered Cricket Analytics
          </div>

          <h1 className="text-5xl md:text-6xl lg:text-7xl font-extrabold tracking-tight leading-[1.05]">
            Track every ball.
            <br />
            <span className="text-gradient">Analyze every move.</span>
          </h1>

          <p className="text-lg md:text-xl text-muted-foreground max-w-xl leading-relaxed">
            Upload your match footage and get instant AI-driven insights on ball speed,
            trajectory, bounce, and swing — all in one beautiful dashboard.
          </p>

          <div className="flex flex-wrap gap-4">
            <Button asChild size="lg" className="gradient-primary text-primary-foreground shadow-elegant hover:shadow-glow transition-smooth h-14 px-8 text-base font-semibold">
              <Link to="/upload">
                <Upload className="mr-2 h-5 w-5" /> Upload Video
              </Link>
            </Button>
            <Button asChild size="lg" variant="outline" className="h-14 px-8 text-base font-semibold border-2">
              <Link to="/results">
                <Play className="mr-2 h-5 w-5" /> View Results
              </Link>
            </Button>
          </div>

          {/* Real honest stats */}
          <div className="flex gap-8 pt-4">
            {[
              { v: "YOLO11",  l: "Detection Model" },
              { v: "Kalman",  l: "Ball Tracking" },
              { v: "6 Zones", l: "Length Classes" },
            ].map((s) => (
              <div key={s.l}>
                <div className="text-2xl md:text-3xl font-bold text-gradient">{s.v}</div>
                <div className="text-sm text-muted-foreground">{s.l}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="relative animate-scale-in">
          <div className="absolute inset-0 gradient-accent rounded-3xl blur-3xl opacity-30 animate-pulse-glow" />
          <div className="relative rounded-3xl overflow-hidden border border-border shadow-elegant">
            <img src={heroImg} alt="Cricket ball tracking visualization"
              className="w-full h-auto" />
          </div>
        </div>
      </div>
    </section>

    {/* Features */}
    <section className="container py-20 space-y-12">
      <div className="text-center space-y-4">
        <h2 className="text-3xl md:text-4xl font-bold tracking-tight">
          How it <span className="text-gradient">works</span>
        </h2>
        <p className="text-muted-foreground max-w-2xl mx-auto">
          Upload any cricket match video. Our AI pipeline detects, tracks, and analyses
          every delivery automatically.
        </p>
      </div>

      <div className="grid md:grid-cols-3 gap-8">
        {features.map(({ icon: Icon, title, desc }) => (
          <div key={title}
            className="rounded-2xl gradient-card border border-border shadow-soft p-8 space-y-4 hover:shadow-elegant transition-smooth hover:-translate-y-1">
            <div className="h-12 w-12 rounded-xl gradient-primary flex items-center justify-center shadow-glow">
              <Icon className="h-6 w-6 text-primary-foreground" />
            </div>
            <h3 className="text-xl font-bold">{title}</h3>
            <p className="text-muted-foreground leading-relaxed">{desc}</p>
          </div>
        ))}
      </div>

      <div className="flex justify-center pt-4">
        <Button asChild size="lg" className="gradient-primary text-primary-foreground shadow-elegant h-14 px-10 text-base font-semibold">
          <Link to="/upload">
            Get Started <ArrowRight className="ml-2 h-5 w-5" />
          </Link>
        </Button>
      </div>
    </section>
  </div>
);

export default Index;
