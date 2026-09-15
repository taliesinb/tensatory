// Colormap interval selection: an interval slider (`.isl.cmap`, drawn as a box) whose background IS the colormap,
// plus three modes that are part of the selection's value (a renderer needs them):
//   span  "stretch" | "full"   included region: colormap compressed to exactly [lo, hi], or the ordinary colormap
//                              over the whole range (the interval then only masks / clips outside)
//   low   "clip" | "mask"      region below lo: saturated to the boundary colour, or not drawn at all
//   high  "clip" | "mask"      region above hi: likewise
// Painting:  included region  stretch -> f(0..1) over [lo, hi];  full -> f(a..b) (the slice of the ordinary map)
//            excluded region  clip -> flat boundary colour (f(0) / f(1) when stretched, f(a) / f(b) when full)
//                             mask -> grey barber-pole hatch
//   half (lo, null): open towards max (no "high" region);  none: the ordinary map over the whole range.
// The widget itself knows nothing about colours; it reports single clicks as 'modetoggle' events ("span" | "low" | "high")
// and this module flips the corresponding mode, repaints, and fires 'change'. Initial modes: data-span / data-low / data-high.
import { type Colormap, toCss } from "../../viewer/src/colormap";
import type { IntervalEl } from "./interval";

export type SpanMode = "stretch" | "full";
export type EdgeMode = "clip" | "mask";
export interface CmapIntervalEl extends IntervalEl {
  span: SpanMode;
  low: EdgeMode;
  high: EdgeMode;
  colormap: Colormap;
}


/** fractions of the bar covered by the interval (open ends run to the range's ends) */
export function intervalFractions(el: IntervalEl): [number, number] {
  const span = el.max - el.min;
  return [((el.lo ?? el.min) - el.min) / span, ((el.hi ?? el.max) - el.min) / span];
}

/** the colour layer: gradient over [a, b] (stretched or the [a, b] slice), flat boundary colours outside */
export function colourGradient(f: Colormap, a: number, b: number, span: SpanMode, n = 24): string {
  const pc = (t: number) => `${(t * 100).toFixed(2)}%`;
  const at = (u: number) => toCss(f(span === "stretch" ? u : a + (b - a) * u)); // u in 0..1 along the included region
  if (b <= a) { // a point: hard step
    const c0 = at(0), c1 = at(1);
    return `linear-gradient(90deg, ${c0} 0%, ${c0} ${pc(a)}, ${c1} ${pc(a)}, ${c1} 100%)`;
  }
  const stops: string[] = [];
  if (a > 0) stops.push(`${at(0)} 0%`);
  for (let i = 0; i <= n; i++) stops.push(`${at(i / n)} ${pc(a + ((b - a) * i) / n)}`);
  if (b < 1) stops.push(`${at(1)} 100%`);
  return `linear-gradient(90deg, ${stops.join(",")})`;
}

/** the bar's CSS `background`: the colour layer (masked sides are covered by the .hatch children) */
export function cmapBackground(el: CmapIntervalEl): string {
  const [a, b] = intervalFractions(el);
  return colourGradient(el.colormap, a, b, el.span);
}

/** value -> colormap parameter under the current selection, or null where the value is masked out */
export function paramOf(el: CmapIntervalEl, v: number): number | null {
  const lo = el.lo ?? el.min, hi = el.hi ?? el.max;
  if (v < lo && el.lo !== null && el.low === "mask") return null;
  if (v > hi && el.hi !== null && el.high === "mask") return null;
  if (el.span === "full") { const t = (Math.min(hi, Math.max(lo, v)) - el.min) / (el.max - el.min); return Math.min(1, Math.max(0, t)); }
  return hi <= lo ? (v < lo ? 0 : 1) : Math.min(1, Math.max(0, (v - lo) / (hi - lo)));
}

/** upgrade an interval slider to a colormap interval: modes, colormap, live repaint, click toggles */
export function makeCmapInterval(el0: IntervalEl, colormap: Colormap): CmapIntervalEl {
  const el = el0 as CmapIntervalEl;
  let span: SpanMode = el.dataset.span === "full" ? "full" : "stretch";
  let low: EdgeMode = el.dataset.low === "mask" ? "mask" : "clip";
  let high: EdgeMode = el.dataset.high === "mask" ? "mask" : "clip";
  let f = colormap;
  // masked sides: two children carrying a fixed 8px hatch tile anchored to the bar's outer edge (so the stripes
  // stay put while the interval moves); a background layer sized to the region would re-phase with its width
  const hatch = (side: "below" | "above") => { const d = document.createElement("div"); d.className = `hatch ${side}`; el.prepend(d); return d; }; // first children: under the frame and handles
  const hatchLo = hatch("below"), hatchHi = hatch("above");
  const paint = () => {
    el.style.background = cmapBackground(el);
    const [a, b] = intervalFractions(el);
    const showLo = el.lo !== null && low === "mask" && a > 0, showHi = el.hi !== null && high === "mask" && b < 1;
    hatchLo.style.display = showLo ? "block" : "none"; hatchLo.style.width = `${(a * 100).toFixed(3)}%`;
    hatchHi.style.display = showHi ? "block" : "none"; hatchHi.style.left = `${(b * 100).toFixed(3)}%`;
  };
  Object.defineProperty(el, "span", { get: () => span, set: (v: SpanMode) => { span = v; paint(); } });
  Object.defineProperty(el, "low", { get: () => low, set: (v: EdgeMode) => { low = v; paint(); } });
  Object.defineProperty(el, "high", { get: () => high, set: (v: EdgeMode) => { high = v; paint(); } });
  Object.defineProperty(el, "colormap", { get: () => f, set: (g: Colormap) => { f = g; paint(); } });
  el.addEventListener("input", paint); el.addEventListener("change", paint);
  el.addEventListener("modetoggle", (e) => {
    const what = (e as CustomEvent<"span" | "low" | "high">).detail;
    if (what === "span") span = span === "stretch" ? "full" : "stretch";
    else if (what === "low") low = low === "clip" ? "mask" : "clip";
    else high = high === "clip" ? "mask" : "clip";
    paint(); el.dispatchEvent(new Event("change"));
  });
  paint();
  return el;
}
