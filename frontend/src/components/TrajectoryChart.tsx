const TrajectoryChart = () => {
  return (
    <svg
      viewBox="0 0 480 240"
      className="w-full h-auto"
      role="img"
      aria-label="Ball trajectory chart"
      style={{ fontFamily: "inherit" }}
    >
      <defs>
        {/* Trajectory gradient: primary → accent */}
        <linearGradient id="trajGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%"   stopColor="hsl(var(--primary))"  stopOpacity="1" />
          <stop offset="100%" stopColor="hsl(var(--accent))"   stopOpacity="1" />
        </linearGradient>

        {/* Subtle glow filter for the ball path */}
        <filter id="glowSoft" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="2.5" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>

        {/* Ground shadow gradient */}
        <linearGradient id="groundGrad" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor="hsl(var(--border))" stopOpacity="0.5" />
          <stop offset="100%" stopColor="hsl(var(--border))" stopOpacity="0" />
        </linearGradient>

        {/* Clip path to keep elements inside chart area */}
        <clipPath id="chartArea">
          <rect x="40" y="10" width="420" height="190" />
        </clipPath>
      </defs>

      {/* ── Background grid lines (horizontal) ── */}
      {[40, 80, 120, 160].map((y) => (
        <line
          key={y}
          x1="40" y1={y} x2="460" y2={y}
          stroke="hsl(var(--border))"
          strokeWidth="0.6"
          strokeDasharray="3 5"
          opacity="0.5"
        />
      ))}

      {/* ── Pitch ground line ── */}
      <line
        x1="40" y1="200" x2="460" y2="200"
        stroke="hsl(var(--border))"
        strokeWidth="1.5"
        opacity="0.8"
      />

      {/* ── Crease marker lines (batting & bowling) ── */}
      {/* Bowling crease */}
      <line x1="60"  y1="196" x2="60"  y2="204" stroke="hsl(var(--muted-foreground))" strokeWidth="1.5" opacity="0.6" />
      {/* Batting crease */}
      <line x1="440" y1="196" x2="440" y2="204" stroke="hsl(var(--muted-foreground))" strokeWidth="1.5" opacity="0.6" />

      {/* ── Pitch surface fill ── */}
      <rect
        x="60" y="198" width="380" height="6"
        fill="hsl(var(--muted))"
        opacity="0.25"
        rx="1"
      />

      {/* ── Shadow arc under trajectory (depth cue) ── */}
      <path
        d="M 65 200 Q 240 205 435 200"
        fill="none"
        stroke="hsl(var(--primary))"
        strokeWidth="1"
        opacity="0.12"
        strokeLinecap="round"
      />

      {/* ── Main trajectory path ── */}
      {/*
          Cricket side-on: bowler (left, ~shoulder height) →
          ball descends → bounces at ~pitch length 60% mark →
          rises slightly toward batsman (right, ~stump height)
      */}
      <path
        d="M 65 52 C 130 68, 190 150, 240 197 C 290 155, 370 130, 435 118"
        fill="none"
        stroke="url(#trajGrad)"
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        filter="url(#glowSoft)"
        clipPath="url(#chartArea)"
      />

      {/* ── Release point ── */}
      <circle cx="65" cy="52" r="5" fill="hsl(var(--primary))" opacity="0.95" />
      <circle cx="65" cy="52" r="8" fill="hsl(var(--primary))" opacity="0.15" />

      {/* ── Bounce point ── */}
      <circle cx="240" cy="197" r="5" fill="hsl(var(--accent))" opacity="0.95" />
      {/* Static ring — no animation */}
      <circle cx="240" cy="197" r="10" fill="none" stroke="hsl(var(--accent))" strokeWidth="1" opacity="0.35" />

      {/* ── End point (stumps) ── */}
      <circle cx="435" cy="118" r="5" fill="hsl(var(--primary))" opacity="0.95" />
      <circle cx="435" cy="118" r="8" fill="hsl(var(--primary))" opacity="0.15" />

      {/* ── Stump verticals ── */}
      {[-5, 0, 5].map((offset) => (
        <line
          key={offset}
          x1={435 + offset} y1="118"
          x2={435 + offset} y2="200"
          stroke="hsl(var(--muted-foreground))"
          strokeWidth="1.2"
          opacity="0.5"
        />
      ))}
      {/* Bail */}
      <line x1="428" y1="118" x2="442" y2="118" stroke="hsl(var(--muted-foreground))" strokeWidth="1.2" opacity="0.5" />

      {/* ── Height axis label ── */}
      <text x="14" y="56"  fontSize="8" fill="hsl(var(--muted-foreground))" textAnchor="middle" opacity="0.7">High</text>
      <text x="14" y="200" fontSize="8" fill="hsl(var(--muted-foreground))" textAnchor="middle" opacity="0.7">Low</text>

      {/* ── Point labels ── */}
      <text x="65"  y="42"  fontSize="9" fill="hsl(var(--muted-foreground))" textAnchor="middle" fontWeight="500">Release</text>
      <text x="240" y="216" fontSize="9" fill="hsl(var(--accent))"           textAnchor="middle" fontWeight="600">Bounce</text>
      <text x="435" y="108" fontSize="9" fill="hsl(var(--muted-foreground))" textAnchor="middle" fontWeight="500">Stumps</text>

      {/* ── Distance annotation ── */}
      <line x1="60" y1="222" x2="240" y2="222" stroke="hsl(var(--border))" strokeWidth="0.8" markerEnd="none" opacity="0.5" />
      <line x1="240" y1="222" x2="440" y2="222" stroke="hsl(var(--border))" strokeWidth="0.8" opacity="0.5" />
      <text x="150" y="232" fontSize="8" fill="hsl(var(--muted-foreground))" textAnchor="middle" opacity="0.6">~12 m</text>
      <text x="340" y="232" fontSize="8" fill="hsl(var(--muted-foreground))" textAnchor="middle" opacity="0.6">~8 m</text>
    </svg>
  );
};

export default TrajectoryChart;
