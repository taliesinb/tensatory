// Colormap interval selection: which part of a field's value range is drawn, and how.
//
// A Selection lives in the field's codomain PARAMETER space (0..1 along its legend bar, so log / flipped
// codomains and range refinements are handled by the codomain), with two independently nullable ends:
//   full  (lo, hi)   half (lo, null) / (null, hi)   none (null, null) = everything, the plain colormap
// and three modes that are part of the selection's value (renderers need them):
//   span  "stretch" | "full"   included region: colormap compressed to exactly [lo, hi], or the ordinary colormap
//                              over the whole range (the interval then only masks / clips outside it)
//   low   "clip" | "mask"      below lo: saturated to the boundary colour, or NOT DRAWN (barber-pole on the bar)
//   high  "clip" | "mask"      above hi: likewise
// What "not drawn" means per use: colorfield -> transparent raster; I_C / S_C -> that stretch of the line is
// not drawn; I_V -> levels in a masked range are not contoured at all. Bars whose field is not used for colour
// (I_V-only, the S_∇ source) are FIXED-MASK: clip / stretch mean nothing there, excluded regions always mask,
// and clicks do not toggle anything.
//
// `selectParam` is the whole semantics in one function; `lutFor` bakes it into a 256-entry RGBA LUT (alpha 0
// where masked) so the canvas raster, the GPU raster and the GPU lines all use one mechanism.
// `makeCmapInterval` upgrades an interval slider (`.isl.cmap`, drawn as a bracket) into the legend control:
// its background is the colormap painted through the selection, live during drags and previews.
import { type Colormap, toCss } from "./colormap";
import type { IntervalEl } from "./interval";

export type SpanMode = "stretch" | "full";
export type EdgeMode = "clip" | "mask";
export interface Selection { lo: number | null; hi: number | null; span: SpanMode; low: EdgeMode; high: EdgeMode }
export const NO_SELECTION: Selection = { lo: null, hi: null, span: "stretch", low: "clip", high: "clip" };

const clamp01 = (t: number) => Math.min(1, Math.max(0, t));
/** effective edge modes: fixed-mask bars always mask */
const modes = (s: Selection, fixedMask: boolean): [EdgeMode, EdgeMode] => (fixedMask ? ["mask", "mask"] : [s.low, s.high]);

/** is the parameter t (0..1) in a masked (not drawn) region? */
export function isMasked(s: Selection, t: number, fixedMask = false): boolean {
  const [low, high] = modes(s, fixedMask);
  return (s.lo !== null && t < s.lo && low === "mask") || (s.hi !== null && t > s.hi && high === "mask");
}

/** codomain parameter t (0..1) -> colormap parameter under the selection; NaN where masked */
export function selectParam(s: Selection, t: number, fixedMask = false): number {
  if (Number.isNaN(t)) return NaN;
  if (isMasked(s, t, fixedMask)) return NaN;
  const lo = s.lo ?? 0, hi = s.hi ?? 1;
  if (s.span === "full") return clamp01(Math.min(hi, Math.max(lo, t)));
  return hi <= lo ? (t < lo ? 0 : 1) : clamp01((t - lo) / (hi - lo));
}

export const selectionKey = (s: Selection, fixedMask = false): string =>
  `${s.lo ?? ""},${s.hi ?? ""},${s.span === "full" ? "f" : "s"}${fixedMask ? "mm" : `${s.low === "mask" ? "m" : "c"}${s.high === "mask" ? "m" : "c"}`}`;
export const isNoSelection = (s: Selection): boolean => s.lo === null && s.hi === null;

/** normalise a stored / untrusted selection */
export function asSelection(o: unknown): Selection {
  const x = (o ?? {}) as Partial<Record<keyof Selection, unknown>>;
  const end = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? clamp01(v) : null);
  let lo = end(x.lo), hi = end(x.hi);
  if (lo !== null && hi !== null && hi < lo) [lo, hi] = [hi, lo];
  return { lo, hi, span: x.span === "full" ? "full" : "stretch", low: x.low === "mask" ? "mask" : "clip", high: x.high === "mask" ? "mask" : "clip" };
}

/* 256-entry RGBA LUTs through a selection, cached per (colormap, selection) */
const LUTS = new Map<string, Uint8Array>();
const cmapIds = new WeakMap<Colormap, number>();
let nextCmapId = 0;
export function lutFor(f: Colormap, s: Selection, fixedMask = false): Uint8Array {
  let id = cmapIds.get(f); if (id === undefined) cmapIds.set(f, (id = nextCmapId++));
  const key = `${id}|${selectionKey(s, fixedMask)}`;
  let lut = LUTS.get(key);
  if (!lut) {
    lut = new Uint8Array(256 * 4);
    const clipped: Selection = { ...s, low: "clip", high: "clip" };
    for (let i = 0; i < 256; i++) {
      let p = selectParam(s, i / 255, fixedMask), a = 255;
      // masked entries are transparent but keep the clipped colour, so linear LUT sampling never blends towards black
      if (Number.isNaN(p)) { p = selectParam(clipped, i / 255); a = 0; }
      const [r, g, b] = f(p);
      lut[4 * i] = Math.round(r * 255); lut[4 * i + 1] = Math.round(g * 255); lut[4 * i + 2] = Math.round(b * 255); lut[4 * i + 3] = a;
    }
    if (LUTS.size > 64) LUTS.delete(LUTS.keys().next().value!);
    LUTS.set(key, lut);
  }
  return lut;
}

/*******************************************************/
/* the legend control */

export interface CmapIntervalEl extends IntervalEl {
  /** the committed selection (during a drag / preview `shown` is what is painted) */
  readonly selection: Selection;
  readonly shown: Selection;
  colormap: Colormap;
}

/** CSS gradient of the colormap through the selection (the bar's background) */
export function selectionGradient(f: Colormap, s: Selection, fixedMask: boolean, n = 32): string {
  const stops: string[] = [];
  const colourAt = (t: number) => { const p = selectParam(s, t, fixedMask); return Number.isNaN(p) ? null : toCss(f(p)); };
  const push = (t: number) => { const c = colourAt(t); if (c) stops.push(`${c} ${(t * 100).toFixed(2)}%`); };
  // sample evenly plus both sides of each end so clip steps and stretch boundaries stay sharp
  const ts = new Set<number>();
  for (let i = 0; i <= n; i++) ts.add(i / n);
  for (const e of [s.lo, s.hi]) if (e !== null) { ts.add(Math.max(0, e - 1e-4)); ts.add(Math.min(1, e + 1e-4)); }
  for (const t of [...ts].sort((a, b) => a - b)) push(t);
  return stops.length ? `linear-gradient(90deg, ${stops.join(",")})` : "transparent";
}

/**
 * Upgrade an interval slider into the legend control. `data-notoggle` on the element = fixed-mask (no colour use).
 * Fires 'input' (live) / 'change' (committed) like the slider; mode toggles fire 'change'.
 */
export function makeCmapInterval(el0: IntervalEl, colormap: Colormap, initial: Selection): CmapIntervalEl {
  const el = el0 as CmapIntervalEl;
  const fixedMask = el.dataset.notoggle !== undefined;
  let span = initial.span, low = initial.low, high = initial.high, f = colormap;
  el.value = `${initial.lo ?? "null"},${initial.hi ?? "null"}`;
  // masked sides: two children carrying an 8px hatch tile anchored to the bar's outer edge (static stripes)
  const hatch = (side: string) => { const d = document.createElement("div"); d.className = `hatch ${side}`; el.prepend(d); return d; };
  const hatchLo = hatch("below"), hatchHi = hatch("above");
  const shown = (): Selection => ({ lo: el.lo, hi: el.hi, span, low, high });
  const paint = () => {
    const s = shown();
    el.style.background = selectionGradient(f, s, fixedMask);
    const [lm, hm] = modes(s, fixedMask);
    const showLo = s.lo !== null && lm === "mask" && s.lo > 0, showHi = s.hi !== null && hm === "mask" && s.hi < 1;
    hatchLo.style.display = showLo ? "block" : "none"; hatchLo.style.width = `${((s.lo ?? 0) * 100).toFixed(3)}%`;
    hatchHi.style.display = showHi ? "block" : "none"; hatchHi.style.left = `${((s.hi ?? 1) * 100).toFixed(3)}%`;
  };
  Object.defineProperty(el, "shown", { get: shown });
  Object.defineProperty(el, "selection", { get: shown }); // callers read it from 'change' handlers, when shown == committed
  Object.defineProperty(el, "colormap", { get: () => f, set: (g: Colormap) => { f = g; paint(); } });
  el.addEventListener("input", paint); el.addEventListener("change", paint);
  el.addEventListener("modetoggle", (e) => {
    if (fixedMask) return;
    const what = (e as CustomEvent<"span" | "low" | "high">).detail;
    if (what === "span") span = span === "stretch" ? "full" : "stretch";
    else if (what === "low") low = low === "clip" ? "mask" : "clip";
    else high = high === "clip" ? "mask" : "clip";
    paint(); el.dispatchEvent(new Event("change"));
  });
  paint();
  return el;
}
