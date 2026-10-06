import { useEffect, useMemo, useState } from "react";
import { Search, SlidersHorizontal, Inbox } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import BallAnalysisCard, { BallData } from "./BallAnalysisCard";

const PAGE_SIZE = 20;

// ALL possible length values — must stay in sync with trajectory.py LENGTH_ZONES
const ALL_LENGTHS = ["Beamer", "Bouncer", "Short", "Good Length", "Full", "Yorker"];

// ALL possible line values — must stay in sync with trajectory.py LINE_ZONES
const ALL_LINES = ["Wide Leg", "Leg Side", "Middle", "Off Side", "Wide Off"];

interface Props {
  fetcher?: () => Promise<BallData[]>;
  deliveries?: any[];
}

const BallAnalysisSection = ({ fetcher, deliveries: deliveriesProp }: Props) => {
  const [balls, setBalls]           = useState<BallData[] | null>(null);
  const [loading, setLoading]       = useState(true);
  const [search, setSearch]         = useState("");
  const [lineFilter, setLineFilter] = useState("all");
  const [lenFilter, setLenFilter]   = useState("all");
  const [speedRange, setSpeedRange] = useState("all");
  const [visible, setVisible]       = useState(PAGE_SIZE);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    (async () => {
      try {
        let data: BallData[];
        if (deliveriesProp && deliveriesProp.length > 0) {
          data = deliveriesProp.map((d) => ({
            ball_id:       d.ball,
            speed:         d.speed,
            line:          d.line,
            length:        d.length,
            swing:         d.swing,
            release_angle: d.release_angle,
            bounce_angle:  d.bounce_angle,
          }));
        } else if (fetcher) {
          data = await fetcher();
        } else {
          data = [];
        }
        if (!cancelled) setBalls(data);
      } catch {
        if (!cancelled) setBalls([]);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [fetcher, deliveriesProp]);

  // Dynamically build option lists from actual data, falling back to the
  // hard-coded full set so filters are always visible even if some categories
  // are absent from the current result set.
  const lineOpts = useMemo(() => {
    const fromData = [...new Set(balls?.map((b) => b.line).filter(Boolean) as string[])];
    const merged   = [...new Set([...ALL_LINES, ...fromData])];
    return merged;
  }, [balls]);

  const lenOpts = useMemo(() => {
    const fromData = [...new Set(balls?.map((b) => b.length).filter(Boolean) as string[])];
    // Keep canonical order
    const merged   = ALL_LENGTHS.filter((l) => fromData.includes(l) || true);
    return merged;
  }, [balls]);

  const filtered = useMemo(() => {
    if (!balls) return [];
    return balls.filter((b) => {
      if (search.trim()) {
        const q = search.trim().toLowerCase().replace(/^ball\s*/, "");
        if (!String(b.ball_id).includes(q)) return false;
      }
      if (lineFilter !== "all" && b.line !== lineFilter) return false;
      if (lenFilter  !== "all" && b.length !== lenFilter) return false;
      if (speedRange !== "all" && typeof b.speed === "number") {
        const [mn, mx] = speedRange.split("-").map(Number);
        if (b.speed < mn || b.speed > mx) return false;
      }
      return true;
    });
  }, [balls, search, lineFilter, lenFilter, speedRange]);

  const reset = () => {
    setSearch(""); setLineFilter("all"); setLenFilter("all");
    setSpeedRange("all"); setVisible(PAGE_SIZE);
  };

  return (
    <section className="space-y-6">
      {/* Heading */}
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 className="text-2xl md:text-3xl font-bold tracking-tight">
            Ball-by-Ball <span className="text-gradient">Analysis</span>
          </h2>
          <p className="text-sm text-muted-foreground mt-1">
            {loading
              ? "Loading…"
              : `${filtered.length}${balls && filtered.length !== balls.length ? ` of ${balls.length}` : ""} deliveries`}
          </p>
        </div>
      </div>

      {/* Filters */}
      <div className="p-4 rounded-2xl gradient-card border border-border shadow-soft">
        <div className="flex flex-wrap gap-3 items-center">
          <div className="relative flex-1 min-w-[180px]">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input placeholder="Search ball number…" value={search}
              onChange={(e) => { setSearch(e.target.value); setVisible(PAGE_SIZE); }}
              className="pl-9" />
          </div>

          {/* Line filter — all 5 zones */}
          <Select value={lineFilter} onValueChange={(v) => { setLineFilter(v); setVisible(PAGE_SIZE); }}>
            <SelectTrigger className="w-[150px]"><SelectValue placeholder="Line" /></SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All lines</SelectItem>
              {lineOpts.map((l) => <SelectItem key={l} value={l}>{l}</SelectItem>)}
            </SelectContent>
          </Select>

          {/* Length filter — all 6 zones always shown */}
          <Select value={lenFilter} onValueChange={(v) => { setLenFilter(v); setVisible(PAGE_SIZE); }}>
            <SelectTrigger className="w-[150px]"><SelectValue placeholder="Length" /></SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All lengths</SelectItem>
              {lenOpts.map((l) => <SelectItem key={l} value={l}>{l}</SelectItem>)}
            </SelectContent>
          </Select>

          {/* Speed filter */}
          <Select value={speedRange} onValueChange={(v) => { setSpeedRange(v); setVisible(PAGE_SIZE); }}>
            <SelectTrigger className="w-[165px]">
              <SlidersHorizontal className="h-4 w-4 mr-1" />
              <SelectValue placeholder="Speed" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">Any speed</SelectItem>
              <SelectItem value="0-120">{"< 120 km/h"}</SelectItem>
              <SelectItem value="120-130">120–130 km/h</SelectItem>
              <SelectItem value="130-140">130–140 km/h</SelectItem>
              <SelectItem value="140-150">140–150 km/h</SelectItem>
              <SelectItem value="150-200">{"> 150 km/h"}</SelectItem>
            </SelectContent>
          </Select>

          <Button variant="ghost" size="sm" onClick={reset}>Reset</Button>
        </div>
      </div>

      {/* Grid */}
      {loading ? (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
          {Array.from({ length: 10 }).map((_, i) => <Skeleton key={i} className="h-52 rounded-2xl" />)}
        </div>
      ) : filtered.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-16 rounded-2xl gradient-card border border-dashed border-border text-center">
          <Inbox className="h-10 w-10 text-muted-foreground mb-3" />
          <p className="text-base font-semibold">No deliveries found</p>
          <p className="text-sm text-muted-foreground mt-1">Try adjusting your filters.</p>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
            {filtered.slice(0, visible).map((ball, idx) => (
              <div key={ball.ball_id ?? idx}
                className="animate-fade-in"
                style={{ animationDelay: `${(idx % PAGE_SIZE) * 30}ms` }}>
                <BallAnalysisCard data={ball} />
              </div>
            ))}
          </div>

          {visible < filtered.length && (
            <div className="flex justify-center pt-2">
              <Button onClick={() => setVisible(c => c + PAGE_SIZE)}
                className="gradient-primary text-primary-foreground shadow-soft">
                Load More ({filtered.length - visible} remaining)
              </Button>
            </div>
          )}
        </>
      )}
    </section>
  );
};

export default BallAnalysisSection;
