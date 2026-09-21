// Tensatory 2D viewer: wiring between the bundle runtime (@tensatory/core),
// the widgets (widgets.ts / metrics.ts) and the canvas renderer (render2d.ts).

import {
  Box,
  Bundle,
  Codomain,
  Sweep,
  rootKind,
  shortHash,
  signatureOf,
  DenseGrid,
  DenseVectorFieldData,
  computeStats,
  defaultStatsGrid,
  formatReal,
  SymbolicScalarFieldData,
  SymbolicVectorFieldData,
  TensatoryError,
  adjustSpec,
  arrowGlyphs,
  controlRows,
  zoomBoxes,
  sliceSpec,
  sliceable,
  zoomable,
  boxBlur,
  contourField,
  integrateFromSeeds,
  isoContours,
  latticeIn,
  latticePoints,
  planStreamlines,
  streamlineSeeds,
  taubinSmooth,
  noArrays,
  type Adjustments,
  type ArrayResolver,
  type ByteSource,
  type ContourResult,
  type GlyphStyle,
  type Lattice,
  type Polyline,
  type ScalarFieldData,
  type Streamline,
  type StreamlineMode,
  type StreamlinePlan,
  type StreamlineSeeds,
  type VectorFieldData,
} from "@tensatory/core";
import { MAPS, cmap, type Colormap } from "./colormap";
import { NO_SELECTION, type Selection, asSelection, isMasked, isNoSelection, lutFor, makeCmapInterval, selectParam, selectionKey } from "./cmapInterval";
import { makeIntervalSlider, type IntervalEl } from "./interval";
import { bootPhase, installLogCapture, showError, status, statusAwaitingPaint } from "./log";
import { MetricsTable, NONE, type MetricsRow, type Sel } from "./metrics";
import type { Curve, Manifold, PointSet } from "@tensatory/core";
import type { BundleSpec } from "@tensatory/schema";
import { Renderer2D, type LineLayer, type Scene, type TriangleLayer } from "./render2d";
import { Sampler, type Values } from "./sampler";
import { GpuGeometry } from "./gpuGeometry";
import { FusedGeometry } from "./gpuFused";
import { View3D, type CropRange, type Use3 } from "./view3d";
import { AutoRes, ladder, type FrameReport, type Tier } from "./autores";
import { Cache, uidOf, type MemoryUser } from "./cache";
import { Recolour } from "./recolour";
import { ControlsPane } from "./controls";
import { CurvesPane, curveOn, curveRange, type CurveDrawable, type CurveOpts } from "./curvesPane";
import { RecordPane } from "./recordPane";
import { bindInfoIcon, installInfoModal, optionText } from "./info";
import { GpuRenderer, type Camera3D, type Quat, quatLook, quatNormalize, gpuStats, gpuTranspilable, packPolylines, packStreamlines, packTriangles, sampleResidentSync, type GpuLineLayer, type GpuScene, type ValueMap, type ColourSource, type GpuBackend, isResidentGrid, SEG_LAYOUT } from "@tensatory/gpu";
import {
  installCollapsiblePanels,
  installTicks,
  fmtNum,
  fmtSlider,
  installTooltips,
  makeChoice,
  makeDiscreteSlider,
  makeSlider,
  syncTicks,
  wheelStepper,
  type ChoiceEl,
  type ValueControl,
} from "./widgets";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

/*******************************************************/
/* widgets */

installLogCapture();
for (const el of document.querySelectorAll<HTMLElement>(".ch:not(.multi)")) makeChoice(el); // before the tooltips: their options carry tips (`.multi`: the slice bar, built per space)
installTooltips();
installInfoModal();
installTicks();
for (const el of document.querySelectorAll<HTMLElement>(".sl")) makeSlider(el);
for (const el of document.querySelectorAll<HTMLElement>(".ds")) makeDiscreteSlider(el);
for (const el of document.querySelectorAll<HTMLElement>(".isl:not(.cmap)")) makeIntervalSlider(el); // the 3D crop ranges

const CHECKS = ["showPoints", "showBox", "showScalar", "smooth", "showIso", "isoAnim", "isoOutline", "isoExact", "showStream", "anim", "showVec"] as const;
const VALUES = ["cropx", "cropy", "cropz", "isoRate", "isoValue", "split", "isoAlpha", "metric", "line", "lines", "slen", "sAlpha", "tail", "ssplit", "sdir", "smode", "vspace", "vglyph", "vAlpha"] as const;
type CheckId = (typeof CHECKS)[number];
type ValueId = (typeof VALUES)[number];
const ui = {
  ...(Object.fromEntries(CHECKS.map((id) => [id, $<HTMLInputElement>(id)])) as Record<CheckId, HTMLInputElement>),
  ...(Object.fromEntries(VALUES.map((id) => [id, $<HTMLElement>(id) as ValueControl])) as Record<ValueId, ValueControl>),
};
const num = (id: ValueId): number | null => { const v = ui[id].value; return v === null ? null : +v; };
/**
 * the streamline seeding mode: the `smode` flipper's label mapped to core's mode plus whether lines are integrated
 * both ways through their seeds (bi-strat: the classic picture) or start at their seeds and run in `dir` only (the
 * others: line starts stay as distributed as the seeds, lines only bunch up where the flow converges)
 */
const STREAM_MODES: Record<string, { mode: StreamlineMode; bidirectional: boolean }> = {
  "bi-strat": { mode: "stratified", bidirectional: true },
  strat: { mode: "stratified", bidirectional: false },
  JL: { mode: "evenly-spaced", bidirectional: false },
  cover: { mode: "coverage", bidirectional: false },
};
const streamMode = (): { mode: StreamlineMode; bidirectional: boolean } => STREAM_MODES[ui.smode.value ?? "strat"] ?? STREAM_MODES.strat!;
/**
 * the integration sign from the `dir` flipper: +1 follows the S∇ field (ascent for a gradient), −1 its inverse
 * (descent). Independent of the ▶/◀ playback direction: a converging field drains descending lines into its sinks,
 * which is not the same picture as ascending lines played backwards.
 */
const streamSign = (): 1 | -1 => (ui.sdir.value === "ascending" ? 1 : -1);
const syncIsoRate = () => { $("isoRate").style.display = ui.isoAnim.checked ? "" : "none"; };
ui.isoAnim.addEventListener("change", syncIsoRate);
syncIsoRate();

/** ▶ ticks: click toggles play keeping the direction; shift-click reverses the direction and makes sure it plays */
function installPlayTick(cb: HTMLInputElement, which: keyof State["dir"]): void {
  const label = document.querySelector<HTMLLabelElement>(`label.tick[for="${cb.id}"]`)!;
  const glyph = label.firstChild as Text;
  const paint = () => { glyph.textContent = state.dir[which] < 0 ? "◀" : "▶"; };
  label.addEventListener("click", (e) => {
    if (e.shiftKey) {
      e.preventDefault();
      state.dir[which] = state.dir[which] < 0 ? 1 : -1;
      cb.checked = true;
      cb.dispatchEvent(new Event("change"));
      state.dirty = true;
    }
    paint();
  });
  playPaints.push(paint);
}
const playPaints: (() => void)[] = [];
const syncPlayGlyphs = () => playPaints.forEach((f) => f());
installPlayTick(ui.isoAnim, "iso");
installPlayTick(ui.anim, "stream");

/*******************************************************/
/* slots: what the mappings matrix assigns fields to */

const SLOTS = ["c", "iv", "ic", "sg", "sc", "vg", "vc"] as const;
type Slot = (typeof SLOTS)[number];
const SLOT_HTML: Record<Slot, string> = { c: "C", iv: "I<sub>V</sub>", ic: "I<sub>C</sub>", sg: "S<sub>∇</sub>", sc: "S<sub>C</sub>", vg: "V<sub>∇</sub>", vc: "V<sub>C</sub>" };
const SLOT_TIP: Record<Slot, string> = {
  c: "colorfield: the field painted as a colormapped raster",
  iv: "isoline value: the field whose level sets are drawn",
  ic: "isoline colour",
  sg: "streamline direction: a vector field, or the gradient of a scalar field",
  sc: "streamline colour",
  vg: "vector field glyphs: a vector field, or the gradient of a scalar field, drawn as arrows on a lattice",
  vc: "glyph colour",
};
/** slot order for legend rows and the cursor pane */
const slotIndex = (k: Slot): number => SLOTS.indexOf(k);

interface State {
  /** the bundle in use: the parsed one, or the one rebuilt from `baseSpec` with the Controls pane's `adjust` applied */
  bundle: Bundle | undefined;
  /** the parsed bundle's spec, as loaded */
  baseSpec: BundleSpec | undefined;
  /** the bundle's external arrays, loaded once with it (adjust / slice / zoom rebuilds keep them) */
  arrays: ArrayResolver;
  /** Controls-pane adjustments (per bundle option): reseed salts and scale multipliers by row id */
  adjust: Adjustments;
  /** box zoom exponent per space (the symbolic fields' boxes scaled by BOX_ZOOM^k around their centres; `-` / `=`) */
  boxZoom: Record<string, number>;
  /** committed slice per N-D space (3 < D <= 8): the 2 or 3 dimensions (0-based, ascending) the space is restricted to,
   *  through the manifold's `origin`; absent = the first three. Applied to the base spec before the Bundle is built. */
  slice: Record<string, number[]>;
  /** per curve id: drawn or not, and the shown parameter range (null ends = the curve's own) */
  curves: Record<string, CurveOpts>;
  /** bumped whenever `bundle` is rebuilt from adjustments / zooms (a frame key component: the fields are new objects) */
  revision: number;
  bundleFile: string; // the loaded document (bundle or sweep), for options storage and the URL
  /** the loaded SWEEP (notes/sweeps.md §2), when the document is one; its members are bundles with metadata records */
  sweep: Sweep | undefined;
  /** the sweep member on view (its id), "" for a lone bundle */
  member: string;
  /** the structural signature (hashed) of the bundle on view: spaces + fields; a sweep's options are keyed by it */
  signature: string;
  /** the selected space (manifold id) of the bundle; fields, point sets and the view belong to it */
  space: string;
  sel: Sel; // shown (may be a hover preview)
  lockedSel: Sel;
  /** colormap index per scalar use id */
  maps: Record<string, number>;
  /** colormap interval selection per scalar use id (codomain parameter space); absent = everything */
  intervals: Record<string, Selection>;
  dirty: boolean;
  paused: boolean;
  animClock: number;
  /** animation (playback) directions (+1 forward, -1 backward); shift-click a ▶ to reverse. For streamlines this
   *  moves the particles forward or backward along the drawn lines; the integration direction is the `dir` control */
  dir: { iso: 1 | -1; stream: 1 | -1 };
}
const emptySel = (): Sel => Object.fromEntries(SLOTS.map((k) => [k, NONE]));
const state: State = { bundle: undefined, baseSpec: undefined, arrays: noArrays, adjust: {}, boxZoom: {}, slice: {}, curves: {}, revision: 0, bundleFile: "", sweep: undefined, member: "", signature: "", space: "", sel: emptySel(), lockedSel: emptySel(), maps: {}, intervals: {}, dirty: true, paused: true, animClock: 0, dir: { iso: 1, stream: 1 } };
const canvas = $<HTMLCanvasElement>("gl");
const renderer = new Renderer2D(canvas);
const sampler = new Sampler(() => { state.dirty = true; });
let geometry: GpuGeometry | undefined; // GPU compute with canvas rendering: asynchronous, read back
let fused: FusedGeometry | undefined; // GPU rendering: resident grids and segment sets
let recolour: Recolour | undefined; // progressive exact recolouring of resident-coloured sets (both arms)
const recolourer3d = (): Recolour => (recolour ??= newRecolour(sampler.gpu!));
const newRecolour = (gpu: GpuBackend): Recolour => new Recolour(gpu); // the frame loop polls `pending` every rAF
let gpuRenderer: GpuRenderer | undefined;
type Compute = "cpu" | "gpu"; type Render = "canvas" | "gpu";
const modes: { compute: Compute; render: Render } = { compute: "cpu", render: "canvas" };
const computeCh = $("computeBar") as ChoiceEl, renderCh = $("renderBar") as ChoiceEl;
computeCh.addEventListener("change", () => { modes.compute = computeCh.value as Compute; applyModes(); });
renderCh.addEventListener("change", () => { modes.render = renderCh.value as Render; applyModes(); });

/** apply the compute / render modes: services, canvases, persistence */
function applyModes(): void {
  const gpu = sampler.gpu;
  if (!gpu) { modes.compute = "cpu"; modes.render = "canvas"; }
  sampler.backend = modes.compute === "gpu" && gpu ? "gpu" : "cpu";
  geometry = gpu && modes.compute === "gpu" && modes.render === "canvas" ? (geometry ?? new GpuGeometry(gpu, () => { state.dirty = true; })) : undefined;
  if (gpu && modes.render === "gpu") {
    fused ??= new FusedGeometry(gpu, () => { state.dirty = true; });
    recolour ??= newRecolour(gpu);
    gpuRenderer ??= new GpuRenderer(gpu, $<HTMLCanvasElement>("gpu"));
  } else { fused?.clear(); fused = undefined; }
  document.body.classList.toggle("gpu-render", modes.render === "gpu" || spaceDims() === 3);
  localStorage.setItem("tensatory.modes", JSON.stringify(modes));
  // the choices reflect the modes (the setters fire no events); unavailable options are greyed out with the reason
  const is3 = spaceDims() === 3;
  computeCh.value = modes.compute; computeCh.setDisabled("gpu", gpu ? false : "no WebGPU adapter");
  renderCh.value = is3 ? "gpu" : modes.render;
  renderCh.setDisabled("canvas", is3 ? "3D spaces render with WebGPU only" : false); renderCh.setDisabled("gpu", gpu ? false : "no WebGPU adapter");
  $("pickCompute").textContent = `${sampler.label}${sampler.check ? " — agreement check on (see L)" : ""}`;
  STREAM_CACHE.clear(); PLAN_CACHE.clear(); isoCache = undefined; glyphCache = undefined; // geometry produced by the other backend
  state.dirty = true;
}
let usable = { scalars: [] as string[], vectors: [] as string[] };

/*******************************************************/
/* field uses: a slot resolves a field id to the field itself, the gradient of a
   scalar (for vector slots) or the norm of a vector (for scalar slots) */

interface ScalarUse { id: string; name: string; codomain: Codomain; data: ScalarFieldData }
interface VectorUse { id: string; name: string; data: VectorFieldData }
const useCache = new Map<string, ScalarUse | VectorUse>();

/**
 * Whether sampling `fd` is expensive for the current compute mode: net-backed data (and whatever derives from it)
 * is evaluated on the CPU unless the GPU computes and can transpile the net (gpu/nets.ts), in which case it is as
 * cheap as any symbolic field — exact isolines, full streamline grids and glyph lattices included.
 */
const costly = (fd: ScalarFieldData | VectorFieldData): boolean => fd.costly && !(modes.compute === "gpu" && !!sampler.gpu && gpuTranspilable(fd));
const paren = (name: string) => (/^[\w.²³]+$/.test(name) ? name : `(${name})`);

function useScalar(id: string | null | undefined): ScalarUse | undefined {
  const b = state.bundle;
  if (!id || !b) return undefined;
  if (usable.scalars.includes(id)) {
    const f = b.scalarField(id);
    return { id, name: f.name, codomain: f.codomain, data: f.data };
  }
  if (!usable.vectors.includes(id)) return undefined;
  const key = `norm:${id}`;
  let u = useCache.get(key) as ScalarUse | undefined;
  if (!u) {
    const v = b.vectorField(id);
    // vector norms are heavy-tailed (gradient norms span orders of magnitude and vanish at critical points): log scale
    u = { id: key, name: `|${v.name}|`, codomain: new Codomain({ min: 0, log: "10" }), data: new SymbolicScalarFieldData({ k: "norm", v: { k: "argv", name: "v" } }, v.data.dimCount, { scalars: {}, vectors: { v: v.data } }) };
    useCache.set(key, u);
  }
  return u;
}

function useVector(id: string | null | undefined): VectorUse | undefined {
  const b = state.bundle;
  if (!id || !b) return undefined;
  if (usable.vectors.includes(id)) {
    const v = b.vectorField(id);
    return { id, name: v.name, data: v.data };
  }
  if (!usable.scalars.includes(id)) return undefined;
  const key = `grad:${id}`;
  let u = useCache.get(key) as VectorUse | undefined;
  if (!u) {
    const f = b.scalarField(id);
    u = { id: key, name: `∇${paren(f.name)}`, data: new SymbolicVectorFieldData({ k: "grad", s: { k: "arg", name: "f" } }, f.data.dimCount, { scalars: { f: f.data }, vectors: {} }) };
    useCache.set(key, u);
  }
  return u;
}

const slotScalar = (k: Slot): ScalarUse | undefined => useScalar(state.sel[k]);
const streamsOn = () => ui.showStream.checked && num("lines") !== null;
const streamVector = (): VectorUse | undefined => (streamsOn() ? useVector(state.sel.sg) : undefined);
const glyphsOn = () => ui.showVec.checked;
/** the V∇ field (glyph arrows) when the vector field panel is on */
const glyphVector = (): VectorUse | undefined => (glyphsOn() ? useVector(state.sel.vg) : undefined);

/*******************************************************/
/* per-use value ranges (for colormaps and the value slider) */

const rangeCache = new Map<string, [number, number]>();
const rangePending = new Set<string>();
function rangeFrom(f: ScalarUse, st: { min: number; max: number }, posMin: () => number): [number, number] {
  const cd = f.codomain;
  let lo = Math.max(st.min, cd.min), hi = Math.min(st.max, cd.max);
  if (cd.log && !(lo > 0)) { const m = posMin(); lo = Number.isFinite(m) ? m : 1e-9; }
  if (!(hi > lo)) hi = lo + 1e-9;
  return [lo, hi];
}
/**
 * Value range of a use (for colormaps and the value slider). Sampled data uses core's stats
 * (cheap, often precomputed). Symbolic data on the GPU: a provisional range from a coarse CPU
 * grid is returned at once and replaced by a GPU reduction over the default grid when it lands.
 */
function rangeOf(f: ScalarUse): [number, number] {
  let r = rangeCache.get(f.id);
  if (r) return r;
  const D = f.data.dimCount;
  if (f.data.kind === "symbolic" && (D === 2 || D === 3) && modes.compute === "gpu" && sampler.gpu) {
    // the coarse grid runs on the main thread: 10³ points keep even an exact symbolic curl of a large
    // expression (~1 ms per point) under a second, where core's default 32³ stats grid would freeze the page
    const coarse = new DenseGrid(D === 2 ? [24, 24] : [10, 10, 10], f.data.box);
    const vals = f.data.sampleOn(coarse);
    r = rangeFrom(f, computeStats(vals), () => { let m = Infinity; for (const v of vals) if (v > 0 && v < m) m = v; return m; });
    rangeCache.set(f.id, r);
    if (!rangePending.has(f.id)) {
      rangePending.add(f.id);
      const gpu = sampler.gpu, id = f.id, grid = costly(f.data) ? new DenseGrid([48, 48], f.data.box) : defaultStatsGrid(f.data.box);
      const shared = D === 2 ? fused : undefined; // 2D: the fused geometry owns resident grids; 3D: a temporary buffer
      const resident = shared ? shared.grid(gridKey(f, grid), f.data, grid) : sampleResidentSync(gpu, f.data, grid);
      gpuStats(gpu, resident).then((st) => {
        if (!Number.isFinite(st.min)) return;
        rangeCache.set(id, rangeFrom(f, st, () => st.posMin));
        updateInfo(); isoCache = undefined; state.dirty = true;
      }).catch((e) => console.warn("GPU stats failed:", e)).finally(() => { rangePending.delete(id); if (!shared) resident.destroy(); });
    }
    return r;
  }
  const st = f.data.stats();
  r = rangeFrom(f, st, () => {
    const g = f.data.samplePoints ?? new DenseGrid(new Array<number>(D).fill(D === 3 ? 16 : 64), f.data.box);
    let m = Infinity;
    for (const v of f.data.sampleOn(g)) if (v > 0 && v < m) m = v;
    return m;
  });
  rangeCache.set(f.id, r);
  return r;
}
const paramOf = (f: ScalarUse) => { const [lo, hi] = rangeOf(f); const cd = f.codomain; return (v: number) => (Number.isNaN(v) ? NaN : cd.toParam(Math.min(hi, Math.max(lo, v)), lo, hi)); };
const mapOf = (f: ScalarUse) => cmap(state.maps[f.id] ?? 0);

/*******************************************************/
/* colormap interval selections (see cmapInterval.ts): per use, in codomain parameter space */

const selOf = (f: ScalarUse): Selection => state.intervals[f.id] ?? NO_SELECTION;
/** a use not mapped to any colour slot has a fixed-mask bar: excluded regions are always masked, no clip / stretch */
const fixedMaskOf = (f: ScalarUse): boolean => !colourSlots().some(([, u]) => u.id === f.id);
const selKeyOf = (f: ScalarUse): string => selectionKey(selOf(f), fixedMaskOf(f));
/** the colormap LUT of a colour use, with its selection baked in (alpha 0 = not drawn) */
const lutOf = (f: ScalarUse): Uint8Array => lutFor(mapOf(f), selOf(f), fixedMaskOf(f));
/** codomain parameter -> colormap parameter for a colour use (NaN = not drawn), for the canvas line renderer */
const selectOf = (f: ScalarUse): ((t: number) => number) => { const s = selOf(f), fm = fixedMaskOf(f); return (t) => selectParam(s, t, fm); };

/*******************************************************/
/* the view box and sampling grid */

function selectedUses(): { scalars: ScalarUse[]; vectors: VectorUse[] } {
  const uses = new Map<string, ScalarUse>();
  const add = (u: ScalarUse | undefined) => { if (u) uses.set(u.id, u); };
  if (ui.showScalar.checked) add(slotScalar("c"));
  if (ui.showIso.checked) { add(slotScalar("iv")); add(slotScalar("ic")); }
  const vectors: VectorUse[] = [];
  const sv = streamVector();
  if (sv) { vectors.push(sv); add(slotScalar("sc")); }
  const gv = glyphVector();
  if (gv) { vectors.push(gv); add(slotScalar("vc")); }
  return { scalars: [...uses.values()], vectors };
}

function unionBox(boxes: Box[]): Box {
  if (!boxes.length) return Box.unit(2);
  const a = [Infinity, Infinity], b = [-Infinity, -Infinity];
  for (const bx of boxes) for (let d = 0; d < 2; d++) { a[d] = Math.min(a[d]!, bx.a[d]!); b[d] = Math.max(b[d]!, bx.b[d]!); }
  return new Box(a, b);
}

let viewBox = Box.unit(2);
let viewBoxKey = "";
function currentViewBox(): Box {
  const { scalars, vectors } = selectedUses();
  let boxes = [...scalars.map((f) => f.data.box), ...vectors.map((v) => v.data.box)];
  if (!boxes.length && state.bundle) boxes = usable.scalars.map((id) => state.bundle!.scalarField(id).data.box);
  const box = unionBox(boxes);
  const key = box.intervals.flat().join(",");
  if (key !== viewBoxKey) { viewBoxKey = key; viewBox = box; if (key) fitView(); }
  return viewBox;
}

function squareGrid(box: Box, n: number): DenseGrid {
  const [w, h] = box.size as [number, number];
  const mx = Math.max(w, h) || 1;
  return new DenseGrid([Math.max(2, Math.round((n * w) / mx) || 2), Math.max(2, Math.round((n * h) / mx) || 2)], box);
}

function currentGrid(box: Box): DenseGrid {
  // native: when every selected sampled field shares one grid filling the view box
  const grids = selectedUses().scalars.map((f) => f.data.samplePoints).filter((g): g is DenseGrid => !!g);
  if (grids.length && grids.every((g) => g.equals(grids[0]!, 1e-12)) && grids[0]!.box.equals(box, 1e-12)) return grids[0]!;
  return squareGrid(box, autoRes2.resolution(tier()));
}

/** field values on the grid (NaN outside the field's own box); undefined while the GPU is still computing */
function sampleUse(u: ScalarUse, grid: DenseGrid): Values | undefined {
  return sampler.request(`${u.id}|${grid.size.join("x")}|${grid.box.intervals.flat().join(",")}`, u.data, grid);
}

/*******************************************************/
/* isolines */

/** `split` levels evenly spaced and centred on `value` (codomain parameter space), wrapping past min/max;
 *  levels inside a masked range of the I_V field's interval selection are not contoured at all */
function isoLevelParams(): number[] {
  const c = +ui.isoValue.value!;
  const split = num("split");
  let out: number[] = [];
  if (split === null) out = [c];
  else for (let k = 0; k < split; k++) { let l = c - 0.5 + (k + 0.5) / split; l = ((l % 1) + 1) % 1; out.push(l); }
  const iv = slotScalar("iv");
  if (iv) { const s = selOf(iv); if (!isNoSelection(s)) { const fm = fixedMaskOf(iv); out = out.filter((t) => !isMasked(s, t, fm)); } }
  return out;
}

interface IsoResult { lines: Polyline[]; values?: (Float64Array | undefined)[]; info: string; rough: boolean }
let isoCache: { key: string; result: IsoResult } | undefined;
/** while the level is moving (animation, slider drag) contours are plain marching squares; exact projection follows once it settles */
let isoLastChange = -1e9;
const ISO_SETTLE_MS = 200;
/** whether the iso animation advances the level: ▶ on, not paused, and the panel enabled (a disabled panel's ▶ is inert) */
const isoAnimating = () => ui.isoAnim.checked && ui.showIso.checked && !state.paused;
const isoMoving = () => isoAnimating() || performance.now() - isoLastChange < ISO_SETTLE_MS;

/*******************************************************/
/* adaptive resolution (autores.ts): one controller per arm; the tier follows the isolines' moving / settled state */

const autoRes2 = new AutoRes(ladder(32, 2048), 2, 64);
const autoRes3 = new AutoRes(ladder(16, 256), 3, 16);
const autoRes = (): AutoRes => (spaceDims() === 3 ? autoRes3 : autoRes2);
/** the 2D grid follows the view box and the 3D grid the crop, so a pan / zoom / crop drag recomputes everything: `moving` too */
let viewLastChange = -1e9;
/** the tier this frame uses: `moving` while the levels change (animation, value / split drag) or the view box /
 *  crop is being dragged, else `settled` */
const tier = (): Tier => ((ui.showIso.checked && slotScalar("iv") && isoMoving()) || performance.now() - viewLastChange < ISO_SETTLE_MS ? "moving" : "settled");
/** a glyph in the top-right corner of the viewport whenever the loop acts (↑ / ↓ a tier step, ⟳ a remeasure):
 *  makes it possible to tell a feedback adjustment from any other stutter */
let flashTimer: ReturnType<typeof setTimeout> | undefined;
/** the resolution flash, top right: one of the SVG glyphs in #resFlash (text glyphs drew from different fallback fonts per browser) */
function flash(glyph: "up" | "down" | "remeasure", title: string): void {
  const el = $("resFlash");
  el.dataset.glyph = glyph; el.title = title; el.classList.add("on");
  clearTimeout(flashTimer); flashTimer = setTimeout(() => el.classList.remove("on"), 700);
}
for (const a of [autoRes2, autoRes3]) {
  a.onChange = (_tier, dir) => { flash(dir > 0 ? "up" : "down", a.note); state.dirty = true; saveOptsSoon(); };
  a.onRemeasure = () => { flash("remeasure", "remeasuring"); fused?.redo(); view3d?.redo(); isoCache = undefined; state.dirty = true; };
}
const MB = 2 ** 20;
/** the memory cap (MB), global: `tensatory.memcap`, ?memcap= */
const memcapEl = $("memcap") as ValueControl;
function applyMemcap(): void {
  const mb = +(memcapEl.value ?? 1024) || 1024;
  autoRes2.capBytes = autoRes3.capBytes = mb * MB;
  localStorage.setItem("tensatory.memcap", String(mb));
  state.dirty = true;
}
memcapEl.addEventListener("input", applyMemcap);
/** everything that holds resident memory */
const memoryUsers = (): MemoryUser[] => [sampler, ...(fused ? [fused] : []), ...(view3d ? [view3d] : [])];
function memoryNow(): { total: number; volume: number; surface: number } {
  let volume = 0, surface = 0, cpu = 0;
  for (const u of memoryUsers()) { const m = u.memory(); volume += m.volume; surface += m.surface; cpu += m.cpu; }
  // device buffers are counted by the backend (whatever cache they sit in), JS arrays by their caches
  return { total: (sampler.gpu?.bytesAllocated ?? 0) + cpu, volume, surface };
}
/** trim caches to the cap (never what this frame uses); true when the working set alone still exceeds it */
function governMemory(cap: number): boolean {
  let over = memoryNow().total - cap * 0.9;
  if (over <= 0) return false;
  for (const u of memoryUsers()) { if (over <= 0) break; over -= u.trim(over); }
  return memoryNow().total > cap;
}
/** 12.3k / 3.31M */
const fmtCount = (n: number) => (n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(n >= 1e5 ? 0 : 1)}k` : String(n));
const fmtMB = (b: number) => String(Math.round(b / MB));
// console access for debugging: tensatory.memory(), tensatory.autoRes()
Object.assign(window, { tensatory: { device: () => sampler.gpu?.device, backend: () => sampler.gpu, view3d: () => view3d, fused: () => fused, memory: () => ({ ...memoryNow(), device: sampler.gpu?.bytesAllocated, users: memoryUsers().map((u) => ({ name: u.constructor.name, ...u.memory(), all: (u as unknown as { debug?: () => unknown }).debug?.() })) }), autoRes } });
function isolines(grid: DenseGrid): IsoResult | undefined {
  const f = slotScalar("iv");
  if (!f || !ui.showIso.checked) return undefined;
  const metric = num("metric"), line = num("line") ?? 0;
  const tol = 0.25 * renderer.worldPerPixel;
  const ic = slotScalar("ic");
  const exactWanted = f.data.kind === "symbolic" && !costly(f.data) && metric === null; // costly (net) fields: marching squares only
  let rough = isoMoving() && exactWanted;
  const gridKey = `${grid.size.join("x")}|${viewBoxKey}`;
  const key = [f.id, gridKey, metric, line, ui.isoValue.value, num("split"), selKeyOf(f), tol.toExponential(2), ic?.id ?? "", rough].join("|");
  if (isoCache?.key === key) return isoCache.result;
  const raw = sampleUse(f, grid);
  if (!raw) return undefined; // still sampling
  const t0 = performance.now();
  const values = metric === null ? raw : boxBlur(grid, raw, metric);
  const [lo, hi] = rangeOf(f);
  const lines: Polyline[] = [];
  const lineLevels: number[] = []; // the level of each line (its colour when I_C = I_V)
  let method = "linear", vertices = 0, maxRes = 0;
  const levels = isoLevelParams().map((t) => f.codomain.fromParam(t, lo, hi));
  // exact lines on the GPU: request every level; while any is still computing, show rough lines this frame
  let gpuResults: ContourResult[] | undefined;
  if (exactWanted && !rough && geometry) {
    gpuResults = [];
    for (const level of levels) {
      const r = geometry.contours(`${f.id}|${gridKey}|${level}|${tol.toExponential(2)}`, `${f.id}|${gridKey}`, f.data, grid, values, level, tol);
      if (!r) { gpuResults = undefined; rough = true; break; }
      gpuResults.push(r);
    }
  }
  for (const [k, level] of levels.entries()) {
    let ls: Polyline[];
    if (gpuResults) {
      const r = gpuResults[k]!;
      method = r.method; ls = r.lines;
      if (r.maxResidual > maxRes) maxRes = r.maxResidual;
    } else if (rough) {
      ls = isoContours(grid, values, level);
    } else if (metric === null) {
      const r = contourField(f.data, grid, values, level, { tolerance: tol });
      method = r.method; ls = r.lines;
      if (r.method === "linear" && line) ls = ls.map((l) => taubinSmooth(l, line));
      if (r.maxResidual > maxRes) maxRes = r.maxResidual;
    } else {
      ls = isoContours(grid, values, level);
      if (line) ls = ls.map((l) => taubinSmooth(l, line));
    }
    lines.push(...ls);
    for (let i = 0; i < ls.length; i++) lineLevels.push(level);
  }
  for (const l of lines) vertices += l.length / 2;
  let colours: (Float64Array | undefined)[] | undefined;
  if (ic && ic.id === f.id) {
    // the colour is the level of each line: no sampling
    const toParam = paramOf(ic);
    colours = lines.map((l, i) => new Float64Array(l.length / 2).fill(toParam(lineLevels[i] ?? NaN)));
  } else if (ic) {
    const toParam = paramOf(ic);
    colours = lines.map((l) => {
      const out = new Float64Array(l.length / 2);
      for (let i = 0; i < out.length; i++) { const v = ic.data.value([l[2 * i]!, l[2 * i + 1]!]); out[i] = v === undefined ? NaN : toParam(v); } // NaN (outside the colour field): not drawn
      return out;
    });
  }
  const ms = (performance.now() - t0).toFixed(1);
  const info = method === "exact" ? `exact${gpuResults ? " (GPU)" : ""}: ${vertices} vertices, max |f − c| = ${maxRes.toExponential(1)}, ${ms} ms` : `marching squares on ${grid.size.join("×")}: ${vertices} vertices, ${ms} ms`;
  const result: IsoResult = { lines, values: colours, info, rough };
  // a rough result standing in for pending GPU lines is keyed as rough so the exact one replaces it when it lands
  isoCache = { key: [f.id, gridKey, metric, line, ui.isoValue.value, num("split"), selKeyOf(f), tol.toExponential(2), ic?.id ?? "", rough].join("|"), result };
  return result;
}

/*******************************************************/
/* streamlines */

interface StreamSet { lines: Streamline[]; step: number; cell: number; colours?: (Float64Array | undefined)[]; colourKey: string }
const STREAM_CACHE = new Map<string, StreamSet>();

/**
 * Symbolic vector fields are sampled once onto a grid (at the current resolution, over the part of
 * the field's box that is in view) and streamlines are integrated through the bilinear interpolant:
 * O(grid) evaluations instead of O(lines × steps × 4), as the 3D prototype did with its gradient grid.
 */
const SAMPLED_VECTORS = new Map<string, DenseVectorFieldData>();
function integrableVector(v: VectorUse, grid: DenseGrid): VectorFieldData | undefined {
  if (v.data.kind === "sampled") return v.data;
  const box = v.data.box.intersect(grid.box) ?? v.data.box;
  const size = [0, 1].map((d) => Math.max(2, Math.round(box.size[d]! / (grid.spacing[d]! || 1)) + 1));
  const key = `${v.id}|${size.join("x")}|${box.intervals.flat().join(",")}`;
  let dense = SAMPLED_VECTORS.get(key);
  if (!dense) {
    const g = new DenseGrid(size, box);
    const values = sampler.request(`vec:${key}`, v.data, g);
    if (!values) return undefined; // still sampling
    dense = new DenseVectorFieldData(g, values);
    SAMPLED_VECTORS.set(key, dense);
    if (SAMPLED_VECTORS.size > 8) SAMPLED_VECTORS.delete(SAMPLED_VECTORS.keys().next().value!);
  }
  return dense;
}

/**
 * The grid streamlines are measured in — the step is ½ cell, `length` counts steps, `tail` counts cells — and
 * that symbolic vectors are sampled on for integration. Same rule as `currentGrid` applies to the scalar fields
 * (the field's own sample grid for sampled fields, else a FIXED 128 grid over the view box — not the adaptive
 * resolution, so line lengths do not change with the tier) but derived from the vector field ALONE: `currentGrid` switches between
 * a native grid and the 128 fallback depending on which scalar panels are enabled, which used to rescale the
 * streamlines whenever the colourfield or the isolines were toggled.
 */
/** `?isoexact=0` turns the 2D exact (projected) isolines off — benchmarking marching squares against projection */
const ISO_EXACT_2D = new URLSearchParams(location.search).get("isoexact") !== "0";
const STREAM_N = 128;
const STREAM_N_COSTLY = 32; // net-backed fields are evaluated on the CPU: keep the fixed grids small
function streamGrid(v: VectorUse, box: Box): DenseGrid {
  if (v.data.samplePoints) return v.data.samplePoints;
  return squareGrid(box, costly(v.data) ? STREAM_N_COSTLY : STREAM_N);
}

/**
 * Planned seed sets (JL / coverage modes): sequential CPU work over the integrable field, cached per
 * (field, grid, mode, options). The plan carries the seeds with their step budgets (for the GPU kernels,
 * which re-integrate them resident) and the lines themselves (used directly by the CPU / read-back paths).
 */
const PLAN_CACHE = new Map<string, StreamlinePlan>();
function streamPlan(key: string, field: VectorFieldData, opts: { count: number; maxSteps: number; step: number; sign: 1 | -1; box: Box; mode: StreamlineMode; bidirectional: boolean }): StreamlinePlan {
  let plan = PLAN_CACHE.get(key);
  if (!plan) {
    const t0 = performance.now();
    plan = planStreamlines(field, { ...opts, seed: 12345 });
    console.log(`streamlines (${opts.mode}): planned ${plan.lines.length} lines, separation ${plan.separation.toExponential(2)}, in ${(performance.now() - t0).toFixed(1)} ms`);
    PLAN_CACHE.set(key, plan);
    if (PLAN_CACHE.size > 8) PLAN_CACHE.delete(PLAN_CACHE.keys().next().value!);
  }
  return plan;
}

let stream: StreamSet | undefined;
function streamlines(view: Box): StreamSet | undefined {
  const v = streamVector();
  const count = num("lines");
  if (!v || count === null) { stream = undefined; return undefined; }
  const grid = streamGrid(v, view);
  const maxSteps = num("slen")!;
  const sign = streamSign(), { mode, bidirectional } = streamMode();
  const cell = Math.min(grid.spacing[0]!, grid.spacing[1]!) || 1e-3;
  const step = 0.5 * cell;
  const field = integrableVector(v, grid);
  if (!field) { stream = undefined; return undefined; } // still sampling
  const box = field.box.intersect(grid.box) ?? field.box;
  const key = [v.id, mode, bidirectional, count, maxSteps, sign, step.toExponential(4), box.intervals.flat().join(",")].join("|");
  let set = STREAM_CACHE.get(key);
  if (!set) {
    let lines: Streamline[] | undefined;
    const iopts = { maxSteps, step, sign, box, bidirectional };
    if (mode !== "stratified") {
      lines = streamPlan(key, field, { count, mode, ...iopts }).lines; // planning integrates: nothing left for the GPU
    } else if (geometry) {
      lines = geometry.streamlines(key, field, streamlineSeeds(box, count, 12345), iopts);
      if (!lines) return stream; // still integrating: keep showing the previous set
    } else {
      const t0 = performance.now();
      lines = integrateFromSeeds(field, streamlineSeeds(box, count, 12345), iopts);
      console.log(`streamlines: ${lines.length} lines in ${(performance.now() - t0).toFixed(1)} ms`);
    }
    set = { lines, step, cell, colourKey: "" };
    STREAM_CACHE.set(key, set);
    if (STREAM_CACHE.size > 16) STREAM_CACHE.delete(STREAM_CACHE.keys().next().value!);
  }
  const s = slotScalar("sc");
  const ck = s?.id ?? "";
  if (set.colourKey !== ck) {
    set.colourKey = ck;
    if (!s) set.colours = undefined;
    else {
      const toParam = paramOf(s);
      set.colours = set.lines.map((l) => {
        const out = new Float64Array(l.points.length / 2);
        for (let i = 0; i < out.length; i++) { const val = s.data.value([l.points[2 * i]!, l.points[2 * i + 1]!]); out[i] = val === undefined ? NaN : toParam(val); }
        return out;
      });
    }
  }
  stream = set;
  return set;
}

/*******************************************************/
/* vector field glyphs (core flow/glyphs.ts): arrows on a lattice whose spacing follows the view */

/** most lattice points a frame samples; the spacing is coarsened until the lattice fits */
const GLYPH_MAX_POINTS = 100_000;
/** ... and for costly (net-backed, CPU-evaluated) fields, whose gradient costs 2D evaluations per point */
const GLYPH_MAX_POINTS_COSTLY = 2_000;
/** 3D lattices are spaced this many times wider than the control says: glyphs at every depth share the screen */
const GLYPH_SPACING_3D = 2;
/** glyphs shorter than this on screen are not drawn (3D: at the camera's target depth): they would only be noise */
const GLYPH_MIN_PX = 4;
const glyphSpacingPx = (): number => num("vspace") ?? 24;
const glyphStyle = (): GlyphStyle => (ui.vglyph.value === "head" || ui.vglyph.value === "triangle" ? ui.vglyph.value : "arrow");

/** the world rectangle the canvas shows (bounding box of its corners: flips and quarter turns keep it a rectangle) */
function visibleWorld(): Box {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  const cs = [renderer.toWorld(0, 0), renderer.toWorld(w, 0), renderer.toWorld(0, h), renderer.toWorld(w, h)];
  return new Box([Math.min(...cs.map((c) => c[0])), Math.min(...cs.map((c) => c[1]))], [Math.max(...cs.map((c) => c[0])), Math.max(...cs.map((c) => c[1]))]);
}

/**
 * The glyph lattice of vector use `v` inside `region` (the view, or the cropped box in 3D) for a view that wants
 * neighbours `target` world units apart (the control's pixels × world-per-pixel). The lattice is FIXED IN SPACE:
 * level k has spacing `longest box side / 2^k`, anchored at the field's box corner, so every finer level contains
 * the coarser one (2Λ ⊂ Λ for the hex and FCC lattices) and the view only picks k — the level whose spacing is
 * nearest `target` in log₂ (so on screen it lies within a factor √2 of the control). Panning, cropping or a zoom
 * that stays within a level moves no glyph; crossing a level tessellates. Only the visible part of the lattice is
 * sampled, so the normalization (the longest vector sampled) packs the most into what is on screen; the level is
 * coarsened while even that exceeds GLYPH_MAX_POINTS. Undefined when the region misses the field.
 */
function glyphLattice(v: VectorUse, region: Box, target: number): Lattice | undefined {
  const fbox = v.data.box, side = Math.max(...fbox.size) || 1;
  const box = fbox.intersect(region);
  if (!box || !(target > 0)) { glyphLevelShown = undefined; return undefined; }
  let k = Math.max(0, Math.round(Math.log2(side / target)));
  const maxPoints = costly(v.data) ? GLYPH_MAX_POINTS_COSTLY : GLYPH_MAX_POINTS;
  for (;;) {
    const lat = latticeIn(box, side / 2 ** k, fbox.a);
    if (lat.pointCount <= maxPoints) { glyphLevelShown = { level: k, spacing: lat.spacing, points: lat.pointCount }; return lat.pointCount ? lat : undefined; }
    if (k === 0) { glyphLevelShown = undefined; return undefined; }
    k--;
  }
}
/** the level the last lattice was built at (for the panel readout) */
let glyphLevelShown: { level: number; spacing: number; points: number } | undefined;
const latticeKey = (l: Lattice): string => `${l.spacing.toExponential(6)}|${l.cosets.map((g) => `${g.size.join("x")}@${g.box.a.map((x) => x.toPrecision(9)).join(",")}`).join(";")}`;

/** arrow / head styles fill `lines` (+ per-vertex colours), the triangle style `triangles` (+ one colour per triangle) */
interface GlyphSet { lines: Float64Array[]; colours?: (Float64Array | undefined)[]; triangles: Float64Array; triColours?: Float64Array; maxNorm: number; key: string }
let glyphCache: GlyphSet | undefined;
/** the last normalizing norm shown in the panel (CPU paths set it directly, the fused path reads it back) */
let glyphMaxShown = NaN;
/**
 * 2D glyphs for the CPU / read-back paths: the field is sampled on every coset of the lattice through the sampler
 * (synchronous on the CPU, asynchronous on the GPU — the previous set stays until the new one lands), then core
 * builds the arrows; colours are the V_C value at each glyph's point.
 */
function glyphs2d(): GlyphSet | undefined {
  const v = glyphVector();
  if (!v) { glyphCache = undefined; return undefined; }
  const lat = glyphLattice(v, visibleWorld(), glyphSpacingPx() * renderer.worldPerPixel);
  if (!lat) { glyphCache = undefined; return undefined; }
  const vc = slotScalar("vc"), style = glyphStyle(), minLength = GLYPH_MIN_PX * renderer.worldPerPixel;
  const key = [v.id, latticeKey(lat), vc?.id ?? "", style, minLength.toPrecision(4)].join("|");
  if (glyphCache?.key === key) return glyphCache;
  const parts = lat.cosets.map((g) => sampler.request(`vec:${v.id}|${g.size.join("x")}|${g.box.intervals.flat().join(",")}`, v.data, g));
  if (parts.some((p) => !p)) return glyphCache; // still sampling
  const vectors = new Float64Array(lat.pointCount * 2);
  let o = 0;
  for (const p of parts) { vectors.set(p!, o); o += p!.length; }
  const points = latticePoints(lat);
  const g = arrowGlyphs(points, vectors, 2, lat.spacing, { style, minLength });
  let colours: (Float64Array | undefined)[] | undefined, triColours: Float64Array | undefined;
  if (vc) {
    const toParam = paramOf(vc);
    const at = (p: number) => { const val = vc.data.value([points[2 * p]!, points[2 * p + 1]!]); return val === undefined ? NaN : toParam(val); };
    colours = g.lines.map((l, k) => new Float64Array(l.length / 2).fill(at(g.point[k]!)));
    triColours = Float64Array.from(g.triPoint, at);
  }
  glyphCache = { lines: g.lines, colours, triangles: g.triangles, triColours, maxNorm: g.maxNorm, key };
  return glyphCache;
}

/*******************************************************/
/* render */

function renderEmpty(): void {
  renderer.render({ box: Box.unit(2), crop: [1, 1], showBox: false, lines: [], pointSets: [] });
  if (modes.render === "gpu" && gpuRenderer) { gpuRenderer.resize(); gpuRenderer.render({ view: renderer.gpuView, clip: Box.unit(2), background: [0x0b / 255, 0x0d / 255, 0x12 / 255], lines: [] }); }
}

function render(): void {
  state.dirty = false;
  fused?.beginFrame();
  if (!state.bundle || !state.space) { renderEmpty(); return; }
  if (spaceDims() === 3) { render3d(); return; }
  const box = currentViewBox();
  const grid = currentGrid(box);
  const scene: Scene = {
    box, crop: [1, 1], showBox: ui.showBox.checked, lines: [],
    pointSets: ui.showPoints.checked ? spacePointSets() : [],
    curves: curveDrawables(),
  };
  const gpuDraw = modes.render === "gpu" && !!fused && !!gpuRenderer;
  const fusedCompute = gpuDraw && modes.compute === "gpu";
  const c = slotScalar("c");
  const cValues = ui.showScalar.checked && c && !fusedCompute ? sampleUse(c, grid) : undefined;
  if (c && cValues && !gpuDraw) {
    const values = cValues;
    scene.raster = { key: `${c.id}|${grid.size.join("x")}|${viewBoxKey}|${state.maps[c.id] ?? 0}|${selKeyOf(c)}`, grid, values, toParam: paramOf(c), lut: lutOf(c), smooth: ui.smooth.checked };
  }
  // fused GPU frames compute isolines / streamlines in renderGpu; otherwise (CPU compute, or GPU compute read back) here
  const iso = fusedCompute ? undefined : isolines(grid);
  const isoField = slotScalar("iv");
  if (iso && !gpuDraw) {
    const alpha = num("isoAlpha") ?? 1;
    const layer: LineLayer = { lines: iso.lines, color: [0.92, 0.92, 0.92], width: 2, alpha };
    const ic = slotScalar("ic");
    if (iso.values && ic) { layer.values = iso.values; layer.cmap = mapOf(ic); layer.select = selectOf(ic); }
    scene.lines.push(layer);
  }
  const st = fusedCompute ? undefined : streamlines(box);
  if (st && !gpuDraw) {
    const layer: LineLayer = { lines: st.lines.map((l) => l.points), color: [1, 1, 1], width: 1.5, alpha: num("sAlpha") ?? 1 };
    const sc = slotScalar("sc");
    if (st.colours && sc) { layer.values = st.colours; layer.cmap = mapOf(sc); layer.select = selectOf(sc); }
    // particles are always drawn at the current phase; ▶ only advances the clock (as in the 3D prototype)
    const tail = num("tail");
    if (tail !== null) layer.particles = { lengths: st.lines.map((l) => l.length), phases: st.lines.map((l) => l.phase), step: st.step, tail: tail * st.cell, split: num("ssplit") ?? 1, travel: state.animClock * 10 * st.cell };
    scene.lines.push(layer);
  }
  const gl = fusedCompute ? undefined : glyphs2d();
  if (gl) {
    glyphMaxShown = gl.maxNorm;
    if (!gpuDraw) {
      const vc = slotScalar("vc"), alpha = num("vAlpha") ?? 1;
      if (gl.triangles.length) {
        const layer: TriangleLayer = { tris: gl.triangles, color: [1, 1, 1], alpha };
        if (gl.triColours && vc) { layer.values = gl.triColours; layer.cmap = mapOf(vc); layer.select = selectOf(vc); }
        (scene.triangles ??= []).push(layer);
      } else {
        const layer: LineLayer = { lines: gl.lines, color: [1, 1, 1], width: 1.5, alpha };
        if (gl.colours && vc) { layer.values = gl.colours; layer.cmap = mapOf(vc); layer.select = selectOf(vc); }
        scene.lines.push(layer);
      }
    }
  }
  if (modes.render === "gpu" && fused && gpuRenderer) renderGpu(grid, box, scene, iso, st, gl);
  else renderer.render(scene);
  updateIsoNotches();
  glyphLabels();

  // labels
  $("resv").textContent = `${grid.size.join("×")}${fused?.info.segments ? ` · ${fmtCount(fused.info.segments)} segs` : ""}${tier() === "moving" && autoRes2.moving !== autoRes2.settled ? ` · settles at ${autoRes2.resolution("settled")}` : ""}`;
  $("isoValuev").textContent = isoField ? fmtIsoValue(isoField) : "—";
  $("splitv").textContent = ui.split.value ?? "—";
  $("isoAlphav").textContent = ui.isoAlpha.value === null ? "—" : fmtSlider(+ui.isoAlpha.value);
  $("metricv").textContent = ui.metric.value ?? "—";
  $("linev").textContent = ui.line.value ?? "—";
  // isoline diagnostics ("exact: N vertices, max |f − c| …" / "marching squares on …"); the #isoInfo row is commented out in index.html
  // $("isoInfo").textContent = iso?.info ?? "";
  $("linesv").textContent = ui.lines.value === null ? "—" : fmtNum(+ui.lines.value);
  $("slenv").textContent = ui.slen.value ?? "";
  $("sAlphav").textContent = ui.sAlpha.value === null ? "—" : fmtSlider(+ui.sAlpha.value);
  $("tailv").textContent = ui.tail.value ?? "";
  $("ssplitv").textContent = ui.ssplit.value ?? "—";
}
/** the `value` readout: the level in the I_V field's codomain, fixed-width (with the codomain's unit) */
const fmtIsoValue = (f: ScalarUse): string => { const s = fmtSlider(f.codomain.fromParam(+ui.isoValue.value!, ...rangeOf(f))); return f.codomain.unit ? `${s} ${f.codomain.unit}` : s; };
/** the vector field panel's readouts (both arms) */
function glyphLabels(): void {
  $("vspacev").textContent = `${ui.vspace.value ?? "—"} px`;
  $("vAlphav").textContent = ui.vAlpha.value === null ? "—" : fmtSlider(+ui.vAlpha.value);
  const gv = glyphVector();
  const lv = gv && glyphLevelShown ? `level ${glyphLevelShown.level} · ${fmt3(glyphLevelShown.spacing)} apart · ${fmtCount(glyphLevelShown.points)} pts in view` : "";
  $("vlevelv").textContent = lv || "—";
  $("vmaxv").textContent = gv && Number.isFinite(glyphMaxShown) && glyphMaxShown > 0 ? `|${gv.name}| = ${fmt3(glyphMaxShown)}` : "—";
}

/*******************************************************/
/* the 3D arm (view3d.ts): isosurfaces of I_V coloured by I_C, WebGPU only */

/* crop ranges: interval sliders whose drags / shift-previews only show a dotted box; the values are committed
   (and the surfaces recomputed) on release */
const CROP_IDS = ["cropx", "cropy", "cropz"] as const;
const cropEl = (id: (typeof CROP_IDS)[number]) => ui[id] as unknown as IntervalEl;
let cropCommitted: CropRange[] = [[null, null], [null, null], [null, null]];
const cropPreviewing = false; // the dotted-outline preview path stays available should a gesture become expensive again
const readCrop = () => { cropCommitted = CROP_IDS.map((id) => [cropEl(id).lo, cropEl(id).hi] as CropRange); };
const fmtCrop = (r: CropRange) => (r[0] === null && r[1] === null ? "—" : `${r[0] === null ? "" : r[0].toFixed(2)}…${r[1] === null ? "" : r[1].toFixed(2)}`);
for (const id of CROP_IDS) {
  const el = cropEl(id);
  // live: the face passes are fixed-size and re-dispatched only, so every crop gesture updates the picture directly
  el.addEventListener("input", () => { readCrop(); viewLastChange = performance.now(); state.dirty = true; }); // the 3D grid follows the crop: a drag is `moving`
  el.addEventListener("change", () => { readCrop(); state.dirty = true; saveOptsSoon(); });
}

let view3d: View3D | undefined;
const gradCache = new Map<string, VectorFieldData>();
function view3dOf(): View3D | undefined {
  const gpu = sampler.gpu;
  if (!gpu) return undefined;
  view3d ??= new View3D({
    gpu,
    canvas: $<HTMLCanvasElement>("gpu"),
    overlay: canvas,
    region: viewRegion, // centred in the free part of the viewport, like the 2D arm
    isoField: () => (ui.showIso.checked ? slotScalar("iv") : undefined),
    colourField: () => slotScalar("ic"),
    gradientOf: (u) => {
      if (u.data.kind !== "symbolic") return undefined;
      let g = gradCache.get(u.id);
      if (!g) gradCache.set(u.id, (g = new SymbolicVectorFieldData({ k: "grad", s: { k: "arg", name: "f" } }, u.data.dimCount, { scalars: { f: u.data }, vectors: {} })));
      return g;
    },
    levels: (u: Use3) => { const f = u as ScalarUse; const [lo, hi] = rangeOf(f); return isoLevelParams().map((t) => f.codomain.fromParam(t, lo, hi)); },
    alpha: () => num("isoAlpha") ?? 1,
    resolution: () => autoRes3.resolution(tier()),
    invalidate: () => { state.dirty = true; },
    blur: () => num("metric"),
    smoothing: () => num("line") ?? 0,
    compute: () => modes.compute,
    costly,
    showIso: () => ui.showIso.checked,
    exact: () => ui.isoExact.checked,
    showOutline: () => ui.isoOutline.checked,
    showPoints: () => ui.showPoints.checked,
    showBox: () => ui.showBox.checked,
    crop: () => cropCommitted,
    cropPreview: () => (cropPreviewing ? CROP_IDS.map((id) => [cropEl(id).lo, cropEl(id).hi] as CropRange) : undefined),
    pointSets: spacePointSets,
    curves: curveDrawables,
    colour: (u: Use3) => { const f = u as ScalarUse; return { map: valueMap(f), lut: lutOf(f), key: selKeyOf(f) }; },
    streamVector: () => streamVector(),
    streamColour: () => slotScalar("sc"),
    recolour: recolourer3d(),
    settled: () => { const ok = !isoMoving() && (autoRes3.pin !== undefined || autoRes3.stable); if (!ok) recolour?.trace(`not settled: ${isoMoving() ? "level moving" : `ladder not holding (${autoRes3.note})`}`); return ok; }, // the level rests and the resolution ladder has stopped climbing
    streamOpts: () => ({ count: num("lines") ?? 0, maxSteps: num("slen")!, sign: streamSign(), ...streamMode(), alpha: num("sAlpha") ?? 1, tail: num("tail"), split: num("ssplit") ?? 1, clock: state.animClock }),
    plan: streamPlan,
    glyphVector: () => glyphVector(),
    glyphColour: () => slotScalar("vc"),
    glyphSpacingPx: () => glyphSpacingPx() * GLYPH_SPACING_3D,
    glyphStyle,
    glyphMinPx: () => GLYPH_MIN_PX,
    glyphLattice: (v, region, spacing) => glyphLattice(v as VectorUse, region, spacing),
  });
  return view3d;
}

function render3d(): void {
  const v = view3dOf();
  if (!v) { renderEmpty(); status("3D spaces need WebGPU"); return; }
  try { v.render(); } catch (e) { showError(e); }
  const isoField = slotScalar("iv");
  $("isoValuev").textContent = isoField ? fmtIsoValue(isoField) : "—";
  $("splitv").textContent = ui.split.value ?? "—";
  $("isoAlphav").textContent = ui.isoAlpha.value === null ? "—" : fmtSlider(+ui.isoAlpha.value);
  const i3 = v.info;
  $("res3v").textContent = `${i3.grid.join("×")}${i3.triangles ? ` · ${fmtCount(i3.triangles)} △` : ""}${tier() === "moving" && autoRes3.moving !== autoRes3.settled ? ` · settles at ${autoRes3.resolution("settled")}` : ""}`;
  $("metricv").textContent = ui.metric.value ?? "—";
  $("linev").textContent = ui.line.value ?? "—";
  $("linesv").textContent = ui.lines.value === null ? "—" : fmtNum(+ui.lines.value);
  $("slenv").textContent = ui.slen.value ?? "";
  $("sAlphav").textContent = ui.sAlpha.value === null ? "—" : fmtSlider(+ui.sAlpha.value);
  $("tailv").textContent = ui.tail.value ?? "";
  $("ssplitv").textContent = ui.ssplit.value ?? "—";
  CROP_IDS.forEach((id, d) => { $(`${id}v`).textContent = fmtCrop(cropPreviewing ? [cropEl(id).lo, cropEl(id).hi] : cropCommitted[d]!); });
  updateIsoNotches();
  glyphMaxShown = v.glyphMaxNorm;
  glyphLabels();
}

/*******************************************************/
/* GPU rendering: resident raster and segment sets (fused when compute is GPU too), canvas overlay for box / points */

function valueMap(u: ScalarUse): ValueMap {
  const [lo, hi] = rangeOf(u);
  return { lo, hi, log: !!u.codomain.log, flip: u.codomain.flip };
}
function gridKey(u: { id: string }, grid: DenseGrid): string { return `${u.id}|${grid.size.join("x")}|${grid.box.intervals.flat().join(",")}`; }

/**
 * What a fused kernel colours its vertices with (gpu/colour.ts): the LEVEL when the colour field is the iso field
 * itself; a resident grid of the field on the dispatch grid — sampled once on the GPU, interpolated at every
 * vertex — when the field is costly (net-backed: an evaluation per vertex was millions of net evaluations); else the
 * field's own program, exact. `key` identifies the source for kernel caches (a resident buffer is bound at build).
 */
function colourSource(c: ScalarUse | undefined, grid: DenseGrid, isoField?: ScalarUse): { src: ColourSource; key: string } {
  if (!c) return { src: undefined, key: "" };
  if (isoField && c.id === isoField.id) return { src: "level", key: "level" };
  if (c.data.costly && fused && modes.compute === "gpu") {
    // the display grid when the field is already resident on it (it is the C field, say), else a capped grid:
    // colour varies smoothly and is interpolated, and a net at 2048² would be 4 M evaluations for a tint
    let g = fused.gridIfResident(gridKey(c, grid));
    if (!g) { const cg = new DenseGrid(grid.size.map((n) => Math.min(n, COLOUR_GRID_MAX_2D)), grid.box); g = fused.grid(gridKey(c, cg), c.data, cg); }
    return { src: g, key: `${c.id}#${uidOf(g)}` };
  }
  return { src: c.data, key: c.id };
}
/** most samples per axis of a costly colour field's resident grid (2D; 3D: View3D.COLOUR_GRID_MAX) */
const COLOUR_GRID_MAX_2D = 256;

function renderGpu(grid: DenseGrid, box: Box, scene2d: Scene, iso: IsoResult | undefined, st: StreamSet | undefined, gl: GlyphSet | undefined): void {
  const F = fused!, R = gpuRenderer!;
  const gs: GpuScene = { view: renderer.gpuView, clip: box, background: [0x0b / 255, 0x0d / 255, 0x12 / 255], lines: [] };
  const fusedCompute = modes.compute === "gpu";
  // raster
  const c = slotScalar("c");
  if (ui.showScalar.checked && c) {
    const key = gridKey(c, grid);
    const values = fusedCompute ? F.grid(key, c.data, grid) : (() => { const v = sampleUse(c, grid); return v ? F.uploadGrid(`cpu:${key}`, grid, v, 1) : undefined; })();
    if (values) gs.raster = { values, box: c.data.box, map: valueMap(c), lut: lutOf(c), smooth: ui.smooth.checked };
  }
  // isolines
  const f = slotScalar("iv"), ic = slotScalar("ic");
  const alpha = num("isoAlpha") ?? 1;
  if (ui.showIso.checked && f) {
    const colour = (ic ? { map: valueMap(ic), lut: lutOf(ic) } : {}) as Partial<GpuLineLayer>;
    if (fusedCompute) {
      const metric = num("metric"), line = num("line") ?? 0;
      const raw = F.grid(gridKey(f, grid), f.data, grid);
      const values = metric === null ? raw : F.blur(gridKey(f, grid), raw, metric);
      // Exact projection of a NET field is latency-bound (~40 ms per level: a lone lane runs Newton's evaluations
      // serially through thread-private memory, whatever the vertex count), so while the level moves such fields show
      // marching squares and turn exact when it settles (as the CPU path always did); analytic fields stay exact
      const exact = f.data.kind === "symbolic" && !costly(f.data) && metric === null && ISO_EXACT_2D && !(f.data.costly && isoMoving());
      const [lo, hi] = rangeOf(f);
      const tol = 0.25 * renderer.worldPerPixel;
      // NOT the colormap selection: it only filters the levels (isoLevelParams) and colours the raster; a kernel is a
      // shader compile (seconds for a net in Safari), and every set re-dispatches on its own level stamp anyway
      const kernelKey = `${f.id}|${gridKey(f, grid)}|${ic?.id ?? ""}|m${metric ?? ""}|${exact ? "exact" : line > 0 ? "smooth" : "ms"}`;
      const icc = colourSource(ic, grid, f);
      const kernelKeyC = `${kernelKey}|${icc.key}`;
      isoLevelParams().forEach((t, k) => {
        const level = f.codomain.fromParam(t, lo, hi);
        const segs = !exact && line > 0
          ? F.smoothedIsolines(kernelKeyC, `${kernelKey}|${k}`, values, icc.src, level, line)
          : F.isolines(kernelKeyC, `${kernelKey}|${k}`, f.data, values, icc.src, level, tol, exact);
        // resident (interpolated) colour: exact colours are written in over the frames once the level rests
        if (ic && isResidentGrid(icc.src) && !isoMoving() && (autoRes2.pin !== undefined || autoRes2.stable) && recolour) { const p = F.recolourProgress(`${kernelKey}|${k}`); if (p) recolour.add(ic.data, ic.id, SEG_LAYOUT, segs.buffer, segs.indirect, p.total, p.progress); }
        gs.lines.push({ segs, width: 2, alpha, color: [0.92, 0.92, 0.92], ...colour });
      });
    } else if (iso) {
      const segs = F.uploadedSegments(`iso|${isoCache?.key ?? ""}`, () => packPolylines(iso.lines, iso.values), false);
      gs.lines.push({ segs, width: 2, alpha, color: [0.92, 0.92, 0.92], ...(iso.values ? colour : {}) });
    }
  }
  // streamlines
  const v = streamVector(), count = num("lines");
  if (v && count !== null) {
    const sc = slotScalar("sc");
    const colour = (sc ? { map: valueMap(sc), lut: lutOf(sc) } : {}) as Partial<GpuLineLayer>;
    const tail = num("tail");
    const particlesIn = (cell: number) => (tail === null ? undefined : { tail: tail * cell, split: num("ssplit") ?? 1, travel: state.animClock * 10 * cell });
    if (fusedCompute) {
      const sgrid = streamGrid(v, box); // NOT `grid`: that one depends on which scalar panels are enabled
      const cell = Math.min(sgrid.spacing[0]!, sgrid.spacing[1]!) || 1e-3;
      const vbox = v.data.box.intersect(sgrid.box) ?? v.data.box;
      const size = [0, 1].map((d) => Math.max(2, Math.round(vbox.size[d]! / (sgrid.spacing[d]! || 1)) + 1));
      const vgrid = new DenseGrid(size, vbox);
      const vectors = F.grid(`vec:${gridKey(v, vgrid)}`, v.data, vgrid);
      const maxSteps = num("slen")!, step = 0.5 * cell, sign = streamSign(), { mode, bidirectional } = streamMode();
      const key = [v.id, gridKey(v, vgrid), mode, bidirectional, count, maxSteps, sign, step.toExponential(4)].join("|");
      const iopts = { maxSteps, step, sign, box: vbox, bidirectional };
      let seeds: StreamlineSeeds | undefined;
      if (mode === "stratified") seeds = streamlineSeeds(vbox, count, 12345);
      else {
        // JL / coverage plans are sequential: planned on the CPU through the same grid (sampled once), then the
        // fused kernel re-integrates the planned seeds within their step budgets — resident, no per-frame cost
        const field = integrableVector(v, sgrid);
        if (field) seeds = streamPlan(key, field, { count, mode, ...iopts }).seeds;
      }
      if (seeds) {
        const scc = colourSource(sc, grid);
        const segs = F.streamlines(`${key}|${scc.key}`, vectors, seeds, iopts, scc.src);
        if (sc && isResidentGrid(scc.src) && recolour) { const p = F.recolourProgress(`${key}|${scc.key}`); if (p) recolour.add(sc.data, sc.id, SEG_LAYOUT, segs.buffer, segs.indirect, p.total, p.progress); }
        gs.lines.push({ segs, width: 1.5, alpha: num("sAlpha") ?? 1, color: [1, 1, 1], particles: particlesIn(cell), ...colour });
      }
    } else if (st) {
      const key = `stream|${v.id}|${st.lines.length}|${st.step}|${st.colourKey}|${st.lines[0]?.points[0] ?? 0}|${st.lines.length && st.lines[st.lines.length - 1]!.points.length}`;
      const segs = F.uploadedSegments(key, () => packStreamlines(st.lines, st.step, st.colours), true);
      gs.lines.push({ segs, width: 1.5, alpha: num("sAlpha") ?? 1, color: [1, 1, 1], particles: particlesIn(st.cell), ...(st.colours ? colour : {}) });
    }
  }
  // vector field glyphs
  const gv = glyphVector();
  if (gv) {
    const vc = slotScalar("vc");
    const colour = (vc ? { map: valueMap(vc), lut: lutOf(vc) } : {}) as Partial<GpuLineLayer>;
    const alpha = num("vAlpha") ?? 1;
    if (fusedCompute) {
      const lat = glyphLattice(gv, visibleWorld(), glyphSpacingPx() * renderer.worldPerPixel);
      if (lat) {
        const style = glyphStyle();
        const vcc = colourSource(vc, grid);
        const segs = F.glyphs(`${gv.id}|${vcc.key}`, gv.data, lat, latticeKey(lat), { style, minLength: GLYPH_MIN_PX * renderer.worldPerPixel }, vcc.src, (m) => { glyphMaxShown = m; glyphLabels(); });
        gs.lines.push({ segs, kind: style === "triangle" ? "triangles" : "lines", width: 1.5, alpha, color: [1, 1, 1], ...colour });
      }
    } else if (gl) {
      const tri = gl.triangles.length > 0;
      const segs = F.uploadedSegments(`glyph|${gl.key}`, () => (tri ? packTriangles(gl.triangles, gl.triColours) : packPolylines(gl.lines, gl.colours)), false);
      gs.lines.push({ segs, kind: tri ? "triangles" : "lines", width: 1.5, alpha, color: [1, 1, 1], ...(gl.colours || gl.triColours ? colour : {}) });
    }
  }
  // a kernel this frame needs is still compiling (async): its dispatch was deferred, so the scene is incomplete —
  // keep the previous image and let the gear spin; `onPipelineReady` re-renders when the compile lands
  if (F.gpu.takeDeferred()) return;
  R.resize();
  R.render(gs);
  renderer.render(scene2d, true); // box, points, labels on the transparent overlay
}

/*******************************************************/
/* legend (#info): one bar per distinct coloured field, listing the slots that use it */

const fmt3 = (v: number) => formatReal(v, 3);
const BLACK: Colormap = () => [0, 0, 0]; // the "colormap" of bars whose field is not used for colour
function centerPoint(): number[] | undefined {
  const ps = spacePointSets().find((p) => p.points.length === 1);
  return ps?.points[0];
}
/** the colour slots currently in use, in display order */
function colourSlots(): [Slot, ScalarUse][] {
  const out: [Slot, ScalarUse][] = [];
  const is2 = spaceDims() === 2;
  const c = slotScalar("c"); if (is2 && ui.showScalar.checked && c) out.push(["c", c]);
  const ic = slotScalar("ic"); if (ui.showIso.checked && ic) out.push(["ic", ic]);
  const sc = slotScalar("sc"); if (streamVector() && sc) out.push(["sc", sc]);
  const vc = slotScalar("vc"); if (glyphVector() && vc) out.push(["vc", vc]);
  return out;
}
const legendUses = new Map<string, ScalarUse>();
/** slots that shape the picture without colouring it: their fields get an all-black bar (min, max, detent, notches, pip) */
function shapeSlots(): [Slot, ScalarUse][] {
  const out: [Slot, ScalarUse][] = [];
  const iv = slotScalar("iv"); if (ui.showIso.checked && iv) out.push(["iv", iv]);
  const sg = state.sel.sg; if (streamsOn() && sg && usable.scalars.includes(sg)) { const u = useScalar(sg); if (u) out.push(["sg", u]); }
  const vg = state.sel.vg; if (glyphsOn() && vg && usable.scalars.includes(vg)) { const u = useScalar(vg); if (u) out.push(["vg", u]); }
  return out;
}
function updateInfo(): void {
  legendUses.clear();
  const groups = new Map<string, { use: ScalarUse; slots: Slot[]; coloured: boolean }>();
  for (const [k, u] of colourSlots()) {
    const g = groups.get(u.id) ?? { use: u, slots: [], coloured: true };
    g.slots.push(k); g.coloured = true; groups.set(u.id, g);
  }
  for (const [k, u] of shapeSlots()) {
    const g = groups.get(u.id) ?? { use: u, slots: [], coloured: false };
    g.slots.push(k); groups.set(u.id, g);
  }
  const c = centerPoint();
  const rows: string[] = [];
  for (const { use: f, slots, coloured } of groups.values()) {
    legendUses.set(f.id, f);
    const [lo, hi] = rangeOf(f), cd = f.codomain;
    const at = c ? f.data.value(c) : undefined;
    const detent = at === undefined ? "" : `<span class="detent" style="left:${(cd.toParam(Math.min(hi, Math.max(lo, at)), lo, hi) * 100).toFixed(1)}%" title="at ${c!.map(fmt3).join(", ")}: ${cd.format(at)}"></span>`;
    const [l, r] = cd.flip ? [hi, lo] : [lo, hi];
    const map = state.maps[f.id] ?? 0;
    // the bar is a colormap interval control (cmapInterval.ts); bars of fields used only for shape are fixed-mask,
    // and a bar used only as the S_∇ / V_∇ source (no colour, no isolines) has nothing to select
    const interval = slots.some((k) => k !== "sg" && k !== "vg");
    const title = coloured
      ? `${MAPS[map % MAPS.length]}${cd.log ? `, log${cd.log}` : ""} — drag on the bar to select the drawn interval; click the name to cycle the colormap`
      : `not used for colour${cd.log ? ` (log${cd.log})` : ""}${interval ? " — drag on the bar: isolines are only drawn at levels inside the selection" : ""}`;
    const slotHtml = SLOTS.filter((k) => slots.includes(k)).map((k) => SLOT_HTML[k]).join(" ");
    const barCls = `bar${coloured ? "" : " plain"}${interval ? " isl cmap" : ""}`;
    const barAttrs = interval ? ` data-min="0" data-max="1" data-step="0.001"${coloured ? "" : " data-notoggle"}` : "";
    rows.push(`<div class="lrow" data-use="${f.id}"><span class="lname${coloured ? " cycle" : ""}" title="${f.name}${coloured ? " — click to cycle the colormap" : ""}"><span class="slot">${slotHtml}</span>${f.name}</span><span class="lval">${cd.format(l)}</span><div class="${barCls}" data-use="${f.id}"${barAttrs} style="background:#000" title="${title}"><span class="notches"></span>${detent}<span class="pip" data-pip="${f.id}"></span></div><span class="lval">${cd.format(r)}</span></div>`);
  }
  const info = $("info");
  info.innerHTML = rows.join("");
  info.style.display = rows.length ? "" : "none";
  for (const el of info.querySelectorAll<HTMLElement>(".lname.cycle")) el.onclick = () => { const id = el.closest<HTMLElement>(".lrow")!.dataset.use!; state.maps[id] = ((state.maps[id] ?? 0) + 1) % MAPS.length; updateInfo(); state.dirty = true; saveOptsSoon(); };
  for (const bar of info.querySelectorAll<HTMLElement>(".bar.isl")) {
    const id = bar.dataset.use!, f = legendUses.get(id)!;
    const w = makeCmapInterval(makeIntervalSlider(bar), bar.classList.contains("plain") ? BLACK : mapOf(f), selOf(f));
    const apply = () => { const sel = w.shown; if (isNoSelection(sel) && sel.span === "stretch" && sel.low === "clip" && sel.high === "clip") delete state.intervals[id]; else state.intervals[id] = sel; state.dirty = true; };
    w.addEventListener("input", apply);
    w.addEventListener("change", () => { apply(); saveOptsSoon(); });
  }
  notchKey = ""; // the bars were rebuilt: the notches must be re-created even if the levels did not change
  updateIsoNotches();
  placeCursorPane();
}

/** white notches on the I_V field's bar at the isoline levels (kept in sync every frame while animating) */
let notchKey = "";
function updateIsoNotches(): void {
  const iv = ui.showIso.checked ? slotScalar("iv") : undefined;
  const params = iv ? isoLevelParams() : [];
  const key = `${iv?.id ?? ""}|${params.map((t) => t.toFixed(4)).join(",")}`;
  if (key === notchKey) return;
  notchKey = key;
  for (const box of document.querySelectorAll<HTMLElement>("#info .notches")) {
    const row = box.closest<HTMLElement>(".lrow");
    const mine = iv && row?.dataset.use === iv.id;
    box.replaceChildren(...(mine ? params : []).map((t) => Object.assign(document.createElement("span"), { className: "notch", style: `left:${(t * 100).toFixed(2)}%` })));
  }
}

/*******************************************************/
/* cursor readout: an untitled pane above the legend; vectors first, then scalars; pips on the legend bars */

function placeCursorPane(): void {
  const info = $("info"), pane = $("cursorPane");
  const infoH = info.style.display === "none" ? 0 : info.offsetHeight + 8;
  pane.style.bottom = `${12 + infoH}px`;
}
function hideCursor(): void {
  $("cursorPane").style.display = "none";
  for (const pip of document.querySelectorAll<HTMLElement>("#info .pip")) pip.style.display = "none";
}
function showCursor(x: number, y: number): void {
  const b = state.bundle;
  if (!b || !viewBox.contains([x, y], 1e-12)) { hideCursor(); return; }
  const esc = (t: string) => t.replace(/&/g, "&amp;").replace(/</g, "&lt;");
  const lines: string[] = [`<b>${esc(currentSpace()?.name ?? "")}:</b> ${fmt3(x)} ${fmt3(y)}`];
  const seenV = new Set<string>();
  for (const vf of [streamVector(), glyphVector()]) {
    if (!vf || seenV.has(vf.id)) continue;
    seenV.add(vf.id);
    const g = vf.data.value([x, y]); if (g) lines.push(`<b>${esc(vf.name)}:</b> ${g.map(fmt3).join(" ")}`);
  }
  const seen = new Set<string>();
  const uses = [...colourSlots(), ...shapeSlots()].sort((a, b) => slotIndex(a[0]) - slotIndex(b[0])).map(([, u]) => u);
  for (const f of uses) {
    if (!f || seen.has(f.id)) continue;
    seen.add(f.id);
    const v = f.data.value([x, y]);
    if (v !== undefined) lines.push(`<b>${esc(f.name)}:</b> ${f.codomain.unit ? f.codomain.format(v) : fmt3(v)}`);
  }
  const pane = $("cursorPane");
  pane.style.display = "block";
  $("cursor").innerHTML = lines.join("<br>");
  placeCursorPane();
  for (const pip of document.querySelectorAll<HTMLElement>("#info .pip")) {
    const f = legendUses.get(pip.dataset.pip!);
    const v = f?.data.value([x, y]);
    if (!f || v === undefined || Number.isNaN(v)) { pip.style.display = "none"; continue; }
    const [lo, hi] = rangeOf(f);
    pip.style.left = `${(f.codomain.toParam(Math.min(hi, Math.max(lo, v)), lo, hi) * 100).toFixed(2)}%`;
    pip.style.display = "block";
  }
}

/*******************************************************/
/* mappings matrix */

const metrics = new MetricsTable({
  body: $("metricSvgBody"),
  slots: [
    { key: "c", label: "C", tip: SLOT_TIP.c, type: "scalar", visible: () => ui.showScalar.checked, toggle: () => ui.showScalar.click(), present: () => spaceDims() === 2 },
    { key: "iv", label: "I", sub: "V", tip: SLOT_TIP.iv, type: "scalar", visible: () => ui.showIso.checked, toggle: () => ui.showIso.click() },
    { key: "ic", label: "I", sub: "C", tip: SLOT_TIP.ic, type: "scalar", visible: () => ui.showIso.checked, toggle: () => ui.showIso.click() },
    { key: "sg", label: "S", sub: "∇", tip: SLOT_TIP.sg, type: "vector", visible: streamsOn, toggle: () => ui.showStream.click() },
    { key: "sc", label: "S", sub: "C", tip: SLOT_TIP.sc, type: "scalar", visible: streamsOn, toggle: () => ui.showStream.click() },
    { key: "vg", label: "V", sub: "∇", tip: SLOT_TIP.vg, type: "vector", visible: glyphsOn, toggle: () => ui.showVec.click() },
    { key: "vc", label: "V", sub: "C", tip: SLOT_TIP.vc, type: "scalar", visible: glyphsOn, toggle: () => ui.showVec.click() },
  ],
  sel: () => state.sel,
  lockedSel: () => state.lockedSel,
  setSel,
  onLayout: () => fitLeftColumn(),
});
/** slots whose change recomputes geometry (isolines / isosurfaces, streamlines, glyphs): switching them pauses the animations */
const GEOMETRY_SLOTS: readonly Slot[] = ["iv", "sg", "vg"];

function setSel(partial: Sel, lock: boolean): void {
  let changed = false, geometry = false;
  for (const [k, id] of Object.entries(partial)) {
    if (lock) { state.lockedSel[k] = id; saveOptsSoon(); }
    if (state.sel[k] === id) continue;
    if (lock && (GEOMETRY_SLOTS as readonly string[]).includes(k)) geometry = true;
    state.sel[k] = id; changed = true;
  }
  if (changed) {
    // a new I_V / S_∇ / V_∇ field means fresh contours / integrations / lattices, for a costly field on the CPU; an
    // animation re-rendering every frame on top of that piles frames up until the page stops responding
    if (geometry) { if (animating()) status("animations paused for the new field (space to resume)"); state.paused = true; }
    if (lock) guardResolution();
    try { updateInfo(); } catch (e) { showError(e); }
    state.dirty = true;
  }
  metrics.refresh();
}

/**
 * A CPU-sampled (costly) field entering the selection must not be sampled at the resolution a transpiled field had
 * climbed to: 2048² × 120 examples on the main thread is minutes, and the controller only judges a frame AFTER it
 * ran. Restart the ladder from the bottom; it ramps back up as far as the budget allows.
 */
function guardResolution(): void {
  const { scalars, vectors } = selectedUses();
  if (![...scalars.map((u) => u.data), ...vectors.map((u) => u.data)].some(costly)) return;
  const a = autoRes();
  if (a.pin === undefined && (a.state().settled > 0 || a.state().moving > 0)) a.restore({ moving: 0, settled: 0 });
}
function matrixRows(): MetricsRow[] {
  const b = state.bundle!;
  return [
    ...usable.scalars.map((id): MetricsRow => ({ id, name: b.scalarField(id).name, kind: "scalar", info: b.scalarField(id).info })),
    ...usable.vectors.map((id): MetricsRow => ({ id, name: b.vectorField(id).name, kind: "vector", info: b.vectorField(id).info })),
  ];
}
function buildMetrics(): void {
  if (!state.bundle) return;
  metrics.build(matrixRows());
  fitLeftColumn();
}

function fitLeftColumn(): void {
  const m = $("metrics");
  $("left").style.maxHeight = `${Math.max(200, window.innerHeight - m.offsetHeight - 44)}px`;
}
window.addEventListener("resize", () => {
  fitLeftColumn();
  if (state.bundle) {
    if (spaceDims() === 3) { if (view3d && !view3d.cameraCustom) view3d.fit(); } else if (!viewCustom) fitView();
  }
  state.dirty = true;
});
installCollapsiblePanels("tensatory.collapsed", fitLeftColumn);

/*******************************************************/
/* options persistence (per bundle) */

let loadingOpts = false, saveTimer: ReturnType<typeof setTimeout> | undefined;
/**
 * where a bundle's options live: `tensatory.opts.<file>`; for a sweep member `tensatory.opts.<sweepFile>#<signature>`,
 * so two members with the same spaces and fields (two seeds, pca vs random directions) share slots, levels, camera
 * and colormaps, and a different architecture has its own
 */
const optsKey = () => (state.bundleFile ? `tensatory.opts.${state.bundleFile}${state.sweep ? `#${state.signature}` : ""}` : null);
/** where a sweep remembers its last member */
const memberKey = () => (state.bundleFile && state.sweep ? `tensatory.member.${state.bundleFile}` : null);
interface SpaceOpts { sel?: Sel; view?: Partial<typeof renderer.view>; dir?: State["dir"]; camera?: Camera3D; res?: { moving: number; settled: number; measured?: boolean } }
/** a saved camera, with the orientation of one saved as yaw / pitch (before the quaternion camera) converted and a malformed `rot` dropped */
function savedCamera(c: Partial<Camera3D> & { yaw?: number; pitch?: number }): Partial<Camera3D> {
  const { yaw, pitch, rot, ...rest } = c;
  if (Array.isArray(rot) && rot.length === 4 && rot.every(Number.isFinite) && Math.hypot(...rot) > 1e-6) return { ...rest, rot: quatNormalize(rot as Quat) };
  if (typeof yaw === "number" && typeof pitch === "number" && Number.isFinite(yaw) && Number.isFinite(pitch)) return { ...rest, rot: quatLook([Math.cos(pitch) * Math.cos(yaw), Math.cos(pitch) * Math.sin(yaw), Math.sin(pitch)]) };
  return rest;
}
interface Opts { ui?: Record<string, unknown>; ui3?: Record<string, unknown>; maps?: Record<string, number>; intervals?: Record<string, unknown>; space?: string; spaces?: Record<string, SpaceOpts>; controls?: Adjustments; boxZoom?: Record<string, number>; slice?: Record<string, number[]>; curves?: Record<string, CurveOpts> }
/**
 * Streamline controls whose good values differ between the arms: saved under `ui` in 2D and `ui3` in 3D, with
 * their own 3D defaults (a volume wants more, fainter lines with several particles each; the HTML `data-value`s
 * are the 2D defaults).
 */
const PER_DIM_VALUES = ["sAlpha", "lines", "ssplit", "tail", "slen"] as const;
const DEFAULTS_3D: Record<(typeof PER_DIM_VALUES)[number], string | null> = { sAlpha: "0.5", lines: "2000", ssplit: "4", tail: "5", slen: "100" };
function readOpts(): Opts {
  const key = optsKey(); const raw = key && localStorage.getItem(key); if (!raw) return {};
  try { const o = JSON.parse(raw) as Opts & { sel?: unknown }; return "sel" in o ? {} : o; } catch { return {}; } // "sel" at the root: pre-space format, ignored
}
function saveOpts(): void {
  const key = optsKey(); if (!key || loadingOpts) return;
  const prev = readOpts();
  const o: Opts = {
    ui: { ...prev.ui, ...Object.fromEntries([...CHECKS.map((id) => [id, ui[id].checked]), ...VALUES.filter((id) => spaceDims() !== 3 || !(PER_DIM_VALUES as readonly string[]).includes(id)).map((id) => [id, ui[id].value])]) },
    ui3: spaceDims() === 3 ? Object.fromEntries(PER_DIM_VALUES.map((id) => [id, ui[id].value])) : prev.ui3,
    maps: state.maps, intervals: state.intervals, space: state.space, controls: state.adjust, boxZoom: state.boxZoom, slice: state.slice, curves: state.curves,
    spaces: { ...prev.spaces, [state.space]: { sel: state.lockedSel, view: viewCustom ? renderer.view : { flipX: renderer.view.flipX, flipY: renderer.view.flipY, rot: renderer.view.rot }, dir: state.dir, res: autoRes().state(), ...(spaceDims() === 3 && view3d?.cameraCustom ? { camera: view3d.camera } : {}) } },
  };
  localStorage.setItem(key, JSON.stringify(o));
}
const saveOptsSoon = () => { clearTimeout(saveTimer); saveTimer = setTimeout(saveOpts, 150); };
const isUsable = (id: string) => usable.scalars.includes(id) || usable.vectors.includes(id);
/** apply the saved options of the bundle (ui, maps, intervals) and of the current space (sel, view, dir) */
function loadOpts(): boolean {
  const o = readOpts();
  const so = o.spaces?.[state.space];
  loadingOpts = true;
  try {
    const dim3 = spaceDims() === 3;
    for (const [id, v] of Object.entries(o.ui ?? {})) {
      if ((CHECKS as readonly string[]).includes(id)) ui[id as CheckId].checked = Boolean(v);
      else if ((VALUES as readonly string[]).includes(id) && !(dim3 && (PER_DIM_VALUES as readonly string[]).includes(id))) ui[id as ValueId].value = v as string | null;
    }
    // the per-dimension controls: this arm's saved values, else its defaults (3D: DEFAULTS_3D; 2D: the HTML data-value)
    for (const id of PER_DIM_VALUES) {
      const saved = dim3 ? o.ui3 : o.ui;
      ui[id].value = (saved && id in saved ? saved[id] : dim3 ? DEFAULTS_3D[id] : ($(id).dataset.value ?? null)) as string | null;
    }
    syncTicks(); syncIsoRate(); readCrop();
    if (o.maps) state.maps = { ...o.maps };
    if (o.intervals) { state.intervals = {}; for (const [id, v] of Object.entries(o.intervals)) { const s = asSelection(v); if (!isNoSelection(s)) state.intervals[id] = s; } }
    if (so?.sel) for (const k of SLOTS) { const id = so.sel[k]; if (id === null || (id && isUsable(id))) state.sel[k] = state.lockedSel[k] = id ?? NONE; }
    if (so?.view) { renderer.view = { ...renderer.view, ...so.view }; viewCustom = so.view.scale !== undefined; }
    // dir.stream used to be the integration sign (now the `sdir` UI value): saves without `sdir` predate that and are not playback directions
    if (so?.dir) state.dir = { iso: so.dir.iso === -1 ? -1 : 1, stream: o.ui?.sdir !== undefined && so.dir.stream === -1 ? -1 : 1 };
    if (so?.camera && view3d) { view3d.camera = { ...view3d.camera, ...savedCamera(so.camera) }; view3d.cameraCustom = true; }
    if (so?.res) autoRes().restore(so.res); // the last good resolutions of this space
    guardResolution(); // ... unless a costly field is selected: those start from the bottom
    syncPlayGlyphs();
  } finally { loadingOpts = false; }
  return !!so;
}

/*******************************************************/
/* bundle loading */

/* spaces: the bundle's 2D and 3D manifolds that carry at least one buildable field */

let buildErrors = new Map<string, Error>();
function spaceList(): Manifold[] {
  const b = state.bundle; if (!b) return [];
  const has = (m: Manifold) => b.fieldIds.some((id) => !buildErrors.has(id) && b.field(id).domain === m);
  return [...b.manifolds.values()].filter((m) => (m.numDims === 2 || m.numDims === 3) && has(m));
}
const currentSpace = (): Manifold | undefined => state.bundle?.manifolds.get(state.space);
const spaceDims = (): number => currentSpace()?.numDims ?? 2;
const spacePointSets = (): PointSet[] => (state.bundle ? [...state.bundle.pointSets.values()].filter((ps) => ps.domain.id === state.space) : []);

/*******************************************************/
/* curves: the bundle's parametrized paths in this space, each drawn over its shown parameter range (curves panel) */

const curvesPane = new CurvesPane($("curvesPanel"));
/** a curve whose slider is being dragged: the shown range follows the pointer before it is committed */
let curvePreview: { id: string; lo: number | null; hi: number | null } | undefined;
curvesPane.onPreview = (id, lo, hi) => { curvePreview = { id, lo, hi }; state.dirty = true; };
const spaceCurves = (): Curve[] => (state.bundle ? state.bundle.curves.filter((c) => c.domain.id === state.space) : []);

const CURVE_MAX_MARKERS = 120; // a densely sampled curve reads as a line, not as beads
function curveDrawables(): CurveDrawable[] {
  const out: CurveDrawable[] = [];
  for (const c of spaceCurves()) {
    const o = state.curves[c.id];
    if (!curveOn(o)) continue;
    const pv = curvePreview?.id === c.id ? curvePreview : undefined;
    const [lo, hi] = pv ? [pv.lo ?? c.data.t0, pv.hi ?? c.data.t1] : curveRange(c, o);
    const D = c.data.dimCount;
    const points = c.data.polyline(lo, hi, 512);
    const markers: CurveDrawable["markers"] = [];
    const ts = c.data.sampleTimes;
    if (ts && ts.length <= CURVE_MAX_MARKERS) {
      const labels = c.spec.data.type === "sampled" ? c.spec.data.labels : undefined;
      for (let i = 0; i < ts.length; i++) { const t = ts[i]!; if (t < lo - 1e-12 || t > hi + 1e-12) continue; const p = c.data.point(t); if (p) markers.push({ p, label: labels?.[i] }); }
    }
    const head = points.length >= D ? Array.from(points.subarray(points.length - D)) : undefined;
    out.push({ id: c.id, points, dimCount: D, markers, head });
  }
  return out;
}

/** a curves-panel change: remember and redraw (nothing is recomputed but the polyline) */
function setCurveOpts(id: string, o: CurveOpts): void {
  const next = { ...state.curves };
  if (o.on !== false && o.lo == null && o.hi == null) delete next[id]; else next[id] = o;
  state.curves = next;
  curvePreview = undefined;
  curvesPane.refresh(state.curves);
  state.dirty = true;
  saveOptsSoon();
}
const spaceSel = $<HTMLSelectElement>("pickSpaceSel");
/** a picker with a single alternative is shown as a plain label (its ⓘ stays); with several, as the select */
function syncPickers(): void {
  for (const [sel, only] of [[pickSel, $("pickBundleOnly")], [spaceSel, $("pickSpaceOnly")]] as const) {
    const single = sel.options.length < 2;
    sel.style.display = single ? "none" : "";
    only.style.display = single ? "" : "none";
    only.textContent = sel.selectedOptions[0]?.textContent ?? "";
  }
}
spaceSel.onchange = () => setSpace(spaceSel.value, true);

/** switch to a space of the current bundle: fields, defaults, saved options and view */
function setSpace(id: string, fromUser: boolean): void {
  const bundle = state.bundle; if (!bundle) return;
  const m = bundle.manifolds.get(id) ?? spaceList()[0]; if (!m) return;
  if (fromUser) saveOpts(); // remember the space we are leaving
  state.space = m.id;
  spaceSel.value = m.id;
  syncPickers();
  bindInfoIcon($("spaceInfo"), m.info);
  syncSliceRow();
  // a fresh space (or bundle) starts PAUSED whatever the saved ▶ ticks say: its first frames build everything from
  // scratch (for a costly field on the CPU), an animation on top piles them up; space resumes
  state.paused = true;
  rangeCache.clear(); useCache.clear(); gradCache.clear(); STREAM_CACHE.clear(); PLAN_CACHE.clear(); SAMPLED_VECTORS.clear(); isoCache = undefined; glyphCache = undefined; viewBoxKey = "";
  usable = {
    scalars: bundle.scalarFieldIds.filter((id) => !buildErrors.has(id) && bundle.scalarField(id).domain === m),
    vectors: bundle.vectorFieldIds.filter((id) => !buildErrors.has(id) && bundle.vectorField(id).domain === m),
  };
  document.body.classList.toggle("dim3", m.numDims === 3);
  curvesPane.build(spaceCurves(), () => state.curves, setCurveOpts);
  $("spaceTitle").textContent = `${m.numDims}D space`;
  $("isoTitle").textContent = m.numDims === 3 ? "isosurfaces" : "isolines";
  $("lineLabel").textContent = m.numDims === 3 ? "surf sm" : "line sm";

  // defaults: colorfield and isoline value on the first scalar field; colour slots none (white lines;
  // a colour slot equal to C would make lines vanish into the raster); streamline / glyph direction from
  // the space's declared dynamical system (`flow`) when it has one, else the first field's exact gradient
  // when the bundle has one, otherwise the symbolic gradient of the field itself
  const first = usable.scalars[0] ?? NONE;
  const exact = first ? bundle.scalarField(first).spec.exactGradient : undefined;
  const flow = m.flow;
  const vec = flow && usable.vectors.includes(flow) ? flow : exact && usable.vectors.includes(exact) ? exact : first;
  const sel: Sel = { c: first, iv: first, ic: NONE, sg: vec, sc: NONE, vg: vec, vc: NONE };
  state.sel = { ...sel }; state.lockedSel = { ...sel };
  state.dir = { iso: 1, stream: 1 }; syncPlayGlyphs();
  viewCustom = false;
  renderer.view = { ...renderer.view, flipX: false, flipY: false, rot: 0 };
  if (m.numDims === 2) currentViewBox(); // establishes the view box (and a default fit) before a saved view may override it
  if (m.numDims === 3) { const v = view3dOf(); if (v) { v.clear(); v.cameraCustom = false; } }
  autoRes().restore({ moving: 0, settled: 0 }); // ramp from the bottom unless this space remembers better
  const hadSaved = loadOpts();
  // a member switch to a signature without saved options for this space: keep what can be kept — the slots whose
  // fields exist here (others fall back to the defaults above), and the view / camera when it is the same space
  if (carry && !hadSaved) {
    for (const k of SLOTS) { const id = carry.sel[k]; if (id === NONE || (id && isUsable(id))) state.sel[k] = state.lockedSel[k] = id; }
    if (carry.space === m.id && carry.dims === m.numDims) {
      if (m.numDims === 2 && carry.view) { renderer.view = { ...carry.view }; viewCustom = carry.viewCustom; }
      if (m.numDims === 3 && carry.camera && view3d) { view3d.camera = { ...carry.camera }; view3d.cameraCustom = carry.cameraCustom; }
    }
  }
  carry = undefined;
  // isolines and streamlines default OFF (a costly field's first frame is then just the raster); a 3D space has no
  // raster, so on its first visit turn the isosurfaces on rather than show an empty box
  if (m.numDims === 3 && !hadSaved && !ui.showIso.checked && !ui.showStream.checked && !ui.showVec.checked) { ui.showIso.checked = true; syncTicks(); }
  // a space with a declared `flow` is a dynamical system: its streamlines run FORWARD in time (following the field,
  // "ascending") on its first visit; the descending default is for gradients of losses. `dir` is a bundle-level
  // control, so a later choice sticks across the bundle's spaces.
  if (m.flow && !hadSaved) ui.sdir.value = "ascending";
  if (m.numDims === 2 && !viewCustom) fitView();
  applyModes();
  $("streamBox").style.display = usable.scalars.length + usable.vectors.length ? "" : "none";
  buildMetrics(); updateInfo();
  $("flipx").classList.toggle("active", renderer.view.flipX);
  $("flipy").classList.toggle("active", renderer.view.flipY);
  const params = new URLSearchParams(location.search); params.set("space", m.id); if (!state.bundleFile.startsWith("local:")) params.set("bundle", state.bundleFile);
  if (state.sweep) params.set("member", state.member); else params.delete("member");
  history.replaceState(null, "", `?${params}`);
  if (fromUser) saveOpts();
  state.dirty = true;
}


/*******************************************************/
/* controls: adjustments (reseed / scale) of the bundle's random directions and arrays */

const controls = new ControlsPane($("controlsPanel"));
const recordPane = new RecordPane($("recordRows"));

// the left stack scrolls within the height the bottom-left column (curves, fields) leaves it
{
  const left = $("left"), bottom = $("bottomLeft");
  const fit = () => { left.style.maxHeight = `${Math.max(120, innerHeight - 24 - bottom.offsetHeight - 8)}px`; };
  new ResizeObserver(fit).observe(bottom);
  window.addEventListener("resize", fit);
  fit();
}

/** the box zoom step: `-` (zoom out) widens every symbolic field's box of the space by this factor, `=` (zoom in) narrows it */
const BOX_ZOOM = 1.5;

/** the default slice of an N-D space: its first three dimensions */
const DEFAULT_SLICE = [0, 1, 2];
/** a valid committed slice of an N-D manifold: 2 or 3 distinct dimensions in range, ascending; else the default */
function sliceOf(spec: BundleSpec, m: string): number[] {
  const D = spec.manifolds?.[m]?.numDims ?? 0;
  const want = state.slice[m];
  const ok = Array.isArray(want) && (want.length === 2 || want.length === 3) && new Set(want).size === want.length && want.every((d) => Number.isInteger(d) && d >= 0 && d < D);
  return ok ? [...want].sort((a, b) => a - b) : DEFAULT_SLICE;
}
/** fields an N-D space lost in its slice (reason by field id), shown with the build errors */
let sliceDropped = new Map<string, string>();

/** the bundle with the current adjustments, slices and box zooms applied (the parsed bundle itself when there are none) */
function adjustedBundle(parsed: Bundle): Bundle {
  const adj = Object.fromEntries(Object.entries(state.adjust).filter(([, a]) => a.seed !== undefined || (a.scale !== undefined && a.scale !== 1)));
  state.adjust = adj;
  state.boxZoom = Object.fromEntries(Object.entries(state.boxZoom).filter(([m, k]) => k !== 0 && Number.isInteger(k) && parsed.manifolds.has(m)));
  let spec = parsed.spec;
  if (Object.keys(adj).length) spec = adjustSpec(spec, adj);
  // N-D spaces become their committed 2D / 3D slice (a spec rewrite: everything downstream sees ordinary fields)
  sliceDropped = new Map();
  for (const m of parsed.manifolds.keys()) {
    if (!sliceable(spec, m)) continue;
    const r = sliceSpec(spec, m, sliceOf(parsed.spec, m), undefined, parsed.arrays);
    for (const [id, why] of Object.entries(r.dropped)) sliceDropped.set(id, why);
    spec = r.spec;
  }
  for (const [m, k] of Object.entries(state.boxZoom)) spec = zoomBoxes(spec, m, Math.pow(BOX_ZOOM, k));
  return spec === parsed.spec ? parsed : new Bundle(spec, parsed.arrays);
}

/** rebuild the bundle from the base spec with the current adjustments / slices / zooms; everything keyed by field id stays */
function rebuildBundle(): void {
  if (!state.baseSpec) return;
  try {
    state.bundle = adjustedBundle(new Bundle(state.baseSpec, state.arrays));
    buildErrors = state.bundle.buildAll();
    showBuildErrors();
    state.revision++;
    clearFieldCaches();
    updateInfo();
  } catch (e) { showError(e); }
  saveOptsSoon();
  state.dirty = true;
}

/** the bundle panel's `errors` row: fields that failed to build, and fields a slice had to leave out */
function showBuildErrors(): void {
  const lines = [...buildErrors].map(([id, e]) => `${id}: ${e.message}`);
  for (const [id, why] of sliceDropped) lines.push(`${id}: not in this slice — ${why}`);
  $("pickErrRow").style.display = lines.length ? "" : "none";
  $("pickErr").textContent = lines.join("\n");
}

/*******************************************************/
/* slice: an N-D space (3 < D <= 8) shown as an axis-aligned 2D / 3D slice through its origin. The bar lists the
   dimensions 1..D; clicking picks / unpicks (blue); the committed slice is tinted; the ✓ appears when the pick has 2 or
   3 dimensions and differs from the committed one, and commits it (a bundle rebuild: `adjustedBundle` slices the spec). */

const sliceBar = $("sliceBar"), sliceTick = $("sliceTick");
/** the dimensions currently picked in the bar (0-based) */
let slicePick = new Set<number>();

/** show / hide the slice row for the current space and paint its segments and tick */
function syncSliceRow(): void {
  const spec = state.baseSpec, m = state.space;
  const show = !!spec && !!m && sliceable(spec, m);
  $("sliceRow").style.display = show ? "" : "none";
  if (!show) return;
  const D = spec.manifolds![m]!.numDims, names = spec.manifolds![m]!.dimNames;
  const committed = sliceOf(spec, m);
  if (sliceBar.childElementCount !== D || sliceBar.dataset.space !== m) {
    sliceBar.dataset.space = m;
    sliceBar.replaceChildren(...Array.from({ length: D }, (_, i) => {
      const seg = document.createElement("div");
      seg.className = "seg"; seg.textContent = String(i + 1);
      seg.dataset.tip = `${names?.[i] ?? `x${i}`}: click to pick / unpick this dimension for the slice`;
      seg.addEventListener("click", () => { if (slicePick.has(i)) slicePick.delete(i); else slicePick.add(i); syncSliceRow(); });
      return seg;
    }));
    installTooltips(sliceBar);
    slicePick = new Set(committed);
  }
  const picked = [...slicePick].sort((a, b) => a - b);
  [...sliceBar.children].forEach((seg, i) => { seg.classList.toggle("on", slicePick.has(i)); seg.classList.toggle("committed", committed.includes(i)); });
  const differs = picked.join(",") !== committed.join(",");
  sliceTick.style.display = (picked.length === 2 || picked.length === 3) && differs ? "" : "none";
}

/** commit the picked dimensions: rebuild the bundle with the new slice and re-enter the space (its arm may change) */
function commitSlice(): void {
  const spec = state.baseSpec, m = state.space;
  if (!spec || !m || !sliceable(spec, m)) return;
  const picked = [...slicePick].sort((a, b) => a - b);
  if (picked.length !== 2 && picked.length !== 3) return;
  state.slice = { ...state.slice, [m]: picked };
  rebuildBundle();
  setSpace(m, true);
  const names = spec.manifolds![m]!.dimNames;
  status(`slice: ${picked.map((d) => names?.[d] ?? `x${d}`).join(", ")} (other coordinates at the space's origin)`);
}
sliceTick.addEventListener("click", commitSlice);

/** `-` / `=`: zoom the symbolic fields' boxes of the current space by BOX_ZOOM (out, dir = 1) / 1 / BOX_ZOOM (in, dir = -1) around their centres */
function stepBoxZoom(dir: 1 | -1): void {
  if (!state.baseSpec || !state.space) return;
  if (!zoomable(state.baseSpec, state.space)) { status("no symbolic fields to zoom in this space (sampled fields keep their grid)"); return; }
  const k = (state.boxZoom[state.space] ?? 0) + dir;
  state.boxZoom = { ...state.boxZoom, [state.space]: k };
  rebuildBundle();
  // the domain changed on purpose: show all of it (a pan / zoom of the view would hide the new margin or leave it empty)
  if (spaceDims() === 3) { if (view3d) { view3d.cameraCustom = false; view3d.fit(); } } else fitView();
  status(k === 0 ? "" : `domain ×${Math.pow(BOX_ZOOM, k).toPrecision(3)} (= zooms in, − out, 0 resets; the colour range and the isoline levels follow the visible domain)`);
}

/**
 * Forget everything derived from the bundle's fields (a rebuild made them new objects): CPU sample and range
 * caches, GPU grids / kernels / sets (their code is cached by the device by text and survives), the recolourers.
 * Selections, view, colormaps and intervals are keyed by field id and stay.
 */
function clearFieldCaches(): void {
  rangeCache.clear(); sampler.clear(); geometry?.clear(); fused?.clear(); view3d?.clear(); recolour?.clear(); gradCache.clear(); useCache.clear(); STREAM_CACHE.clear(); PLAN_CACHE.clear(); SAMPLED_VECTORS.clear(); isoCache = undefined; glyphCache = undefined; viewBoxKey = "";
}

/** a Controls row changed: rebuild the bundle from the base spec with the adjustments, keep everything else */
function applyAdjustment(id: string, a: { seed?: number; scale?: number }): void {
  if (!state.baseSpec) return;
  const next = { ...state.adjust }; if (a.seed === undefined && (a.scale === undefined || a.scale === 1)) delete next[id]; else next[id] = a;
  state.adjust = next;
  rebuildBundle();
}

/** whether an animation is playing: the Controls rows are inert then (a rebuild would stutter it) */
const animating = (): boolean => (isoAnimating() && !!slotScalar("iv")) || (!state.paused && ui.anim.checked && num("lines") !== null && !!streamVector());

/** what a member switch carries into a member whose signature has no saved options yet */
let carry: { sel: Sel; space: string; dims: number; view: typeof renderer.view | undefined; viewCustom: boolean; camera: Camera3D | undefined; cameraCustom: boolean } | undefined;

function setBundle(parsed: Bundle, file: string, wantSpace?: string | null, member = ""): void {
  state.bundleFile = file; state.member = member; state.signature = shortHash(signatureOf(parsed.spec)); state.baseSpec = parsed.spec; state.arrays = parsed.arrays;
  const opts0 = readOpts(); state.adjust = opts0.controls ?? {}; state.boxZoom = opts0.boxZoom ?? {}; state.slice = opts0.slice ?? {}; state.curves = opts0.curves ?? {};
  let bundle: Bundle;
  try { bundle = adjustedBundle(parsed); } catch (e) { showError(e); state.adjust = {}; state.boxZoom = {}; state.slice = {}; try { bundle = adjustedBundle(parsed); } catch { bundle = parsed; } } // stale adjustments (the bundle changed): drop them
  state.bundle = bundle; state.revision++;
  clearFieldCaches(); state.maps = {}; state.intervals = {};
  buildErrors = bundle.buildAll();
  controls.build(controlRows(parsed.spec), () => state.adjust, applyAdjustment);
  // the bundle's ⓘ (summary / details); the loaded bundle knows more than its index entry, so its option's hover follows.
  // For a sweep the `bundle` row is the sweep (its ⓘ); the member and its ⓘ are the record pane's first row.
  const docInfo = state.sweep ? state.sweep.info : bundle.info;
  bindInfoIcon($("bundleInfo"), docInfo);
  const opt = [...pickSel.options].find((o) => o.value === file); if (opt) opt.title = optionText(docInfo);
  recordPane.build(state.sweep, member, bundle.info, (id, why) => void setMember(id, why));
  showBuildErrors();
  const spaces = spaceList();
  spaceSel.replaceChildren(...spaces.map((m) => Object.assign(document.createElement("option"), { value: m.id, textContent: m.name, title: optionText(m.info) })));
  syncPickers();
  const saved = readOpts().space;
  const pick = [wantSpace, saved, spaces.find((m) => m.numDims === 2)?.id, spaces[0]?.id].find((id) => id && spaces.some((m) => m.id === id));
  if (!pick) { usable = { scalars: [], vectors: [] }; state.space = ""; buildMetrics(); updateInfo(); status("no 2D or 3D space with fields"); state.dirty = true; return; }
  setSpace(pick, false);
  status("");
}

/** the sidecar files of a bundle document at `url`: `path` is relative to the document; a 404 is `null` (a missing zarr chunk) */
function fetchSource(url: URL): ByteSource {
  return {
    bytes: async (path) => {
      const res = await fetch(new URL(path, url), { cache: "no-cache" });
      if (res.status === 404) return null;
      if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${path}`);
      return res.arrayBuffer();
    },
  };
}

async function loadBundle(file: string, wantSpace?: string | null, wantMember?: string | null): Promise<void> {
  status(`loading ${file}…`);
  const t0 = performance.now();
  try {
    const url = new URL(`bundles/${file}`, location.href);
    const res = await fetch(url, { cache: "no-cache" });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText} for bundles/${file}`);
    const json = await res.json();
    console.log(`bundle ${file}: fetched in ${(performance.now() - t0).toFixed(0)} ms`);
    if (rootKind(json) === "sweep") {
      // a sweep: its members are fetched (with their sidecars) only when selected; ?member=, else the last one viewed, else the first
      const sweep = Sweep.parse(json, fetchSource(url));
      if (!sweep.memberIds.length) throw new TensatoryError("the sweep has no members");
      state.sweep = sweep; state.bundleFile = file;
      const saved = localStorage.getItem(memberKey()!) ?? undefined;
      const member = [wantMember ?? undefined, saved, sweep.memberIds[0]].find((id): id is string => id !== undefined && sweep.spec.members[id] !== undefined)!;
      await setMember(member, undefined, wantSpace);
      console.log(`sweep ${file}: ready in ${(performance.now() - t0).toFixed(0)} ms (member ${member}, space ${state.space})`);
      return;
    }
    state.sweep = undefined;
    // external arrays (`handle` specs) live beside the document; they are loaded now so the build stays synchronous
    const parsed = await Bundle.load(json, fetchSource(url), { onProgress: (p) => status(`loading ${file}… arrays ${p.done}/${p.total}`) });
    setBundle(parsed, file, wantSpace);
    console.log(`bundle ${file}: ready in ${(performance.now() - t0).toFixed(0)} ms (space ${state.space})`);
  } catch (e) {
    console.error(e);
    status(e instanceof TensatoryError || e instanceof Error ? e.message : String(e));
  }
}

/**
 * Switch the loaded sweep to member `id`: the current options are saved under the current signature first (so a
 * member sharing it finds them), the member's bundle is fetched once (`Sweep.member` caches), then shown like any
 * bundle — the same space when the member has it. `why` is the record pane's account of the switch, for the status line.
 */
async function setMember(id: string, why?: string, wantSpace?: string | null): Promise<void> {
  const sweep = state.sweep; if (!sweep) return;
  if (state.bundle) saveOpts();
  status(`loading member ${id}…`);
  try {
    const b = await sweep.member(id, { onProgress: (p) => status(`loading member ${id}… arrays ${p.done}/${p.total}`) });
    if (state.sweep !== sweep) return; // another document was loaded meanwhile
    if (state.bundle) carry = { sel: { ...state.lockedSel }, space: state.space, dims: spaceDims(), view: { ...renderer.view }, viewCustom, camera: view3d ? { ...view3d.camera } : undefined, cameraCustom: view3d?.cameraCustom ?? false };
    setBundle(b, state.bundleFile, wantSpace ?? state.space, id);
    const mk = memberKey(); if (mk) localStorage.setItem(mk, id);
    status(why ? `${b.name}: ${why}` : "");
  } catch (e) {
    console.error(e);
    carry = undefined;
    status(e instanceof TensatoryError || e instanceof Error ? e.message : String(e));
  }
}

/** bundles/index.json: `summary` mirrors the bundle's own (the hover of a not-yet-loaded alternative) */
let bundleList: { file: string; name?: string; summary?: string }[] = [];
const pickSel = $<HTMLSelectElement>("pickBundle");
function chooseBundle(i: number): void {
  if (!bundleList.length) return;
  const b = bundleList[((i % bundleList.length) + bundleList.length) % bundleList.length]!;
  pickSel.value = b.file;
  void loadBundle(b.file);
}
pickSel.onchange = () => void loadBundle(pickSel.value);
stepOnWheel(pickSel, (dir) => chooseBundle(pickSel.selectedIndex + dir));
stepOnWheel(spaceSel, (dir) => { const n = spaceSel.options.length; if (n < 2) return; spaceSel.selectedIndex = (((spaceSel.selectedIndex + dir) % n) + n) % n; setSpace(spaceSel.value, true); });

/** step a <select> with the wheel or ↑/↓ while hovering it (same one-step-per-gesture wheel handling as the discrete sliders) */
function stepOnWheel(sel: HTMLSelectElement, step: (dir: number) => void): void {
  let over = false;
  sel.addEventListener("pointerenter", () => (over = true)); sel.addEventListener("pointerleave", () => (over = false));
  const wheel = wheelStepper(step);
  sel.addEventListener("wheel", (e) => { if (e.shiftKey) return; wheel(e); }, { passive: false });
  window.addEventListener("keydown", (e) => { if (!over || e.shiftKey) return; if (e.key === "ArrowDown") { e.preventDefault(); step(1); } else if (e.key === "ArrowUp") { e.preventDefault(); step(-1); } });
}
$("uploadBtn").onclick = () => $<HTMLInputElement>("pickFile").click();
/* local bundles: pick the JSON alone, or together with its sidecar files (.bin / .npy / .npz beside it; a zarr store
   cannot be picked as flat files, so handles into one fail on their fields with a clear message) */
$<HTMLInputElement>("pickFile").addEventListener("change", async (ev) => {
  const files = [...((ev.target as HTMLInputElement).files ?? [])]; if (!files.length) return;
  const doc = files.find((f) => /\.json$/i.test(f.name)) ?? files[0]!;
  const sidecars = files.filter((f) => f !== doc);
  const src: ByteSource = { bytes: async (path) => sidecars.find((f) => f.name === path || (f as File & { webkitRelativePath?: string }).webkitRelativePath === path)?.arrayBuffer() ?? null };
  try {
    const json = JSON.parse(await doc.text()) as unknown;
    if (rootKind(json) === "sweep") {
      // a sweep picked locally: inline members work; members by path need their documents among the picked files (flat names only)
      const sweep = Sweep.parse(json, src);
      if (!sweep.memberIds.length) throw new TensatoryError("the sweep has no members");
      state.sweep = sweep; state.bundleFile = `local:${doc.name}`;
      await setMember(sweep.memberIds[0]!);
      return;
    }
    state.sweep = undefined;
    const parsed = await Bundle.load(json, src);
    setBundle(parsed, `local:${doc.name}`);
    status(`${doc.name} (local${sidecars.length ? `, ${sidecars.length} sidecar file${sidecars.length === 1 ? "" : "s"}` : ""})`);
  } catch (e) { console.error(e); status(e instanceof Error ? e.message : String(e)); }
});

/*******************************************************/
/* interaction */

{
  let drag: { x: number; y: number; pan: boolean } | null = null;
  canvas.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY, pan: e.shiftKey || e.button === 2 }; canvas.setPointerCapture(e.pointerId); view3d?.flingCancel(); state.dirty = true; });
  canvas.addEventListener("pointermove", (e) => {
    if (drag && spaceDims() === 3) {
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag = { ...drag, x: e.clientX, y: e.clientY };
      if (view3d) { if (drag.pan) view3d.pan(dx, dy); else { view3d.orbit(dx, dy); view3d.flingTrack(dx, dy, e.timeStamp); } state.dirty = true; }
      return;
    }
    if (drag) { renderer.pan(e.clientX - drag.x, e.clientY - drag.y); drag = { ...drag, x: e.clientX, y: e.clientY }; viewCustom = true; viewLastChange = performance.now(); state.dirty = true; return; }
    if (spaceDims() === 3) return;
    const r = canvas.getBoundingClientRect();
    const [x, y] = renderer.toWorld(e.clientX - r.left, e.clientY - r.top);
    showCursor(x, y);
  });
  canvas.addEventListener("pointerup", (e) => {
    // an orbit released while the pointer is still moving keeps spinning (a click or a drag that came to rest does not)
    if (drag && !drag.pan && spaceDims() === 3 && view3d?.flingRelease(e.timeStamp)) state.dirty = true;
    drag = null; saveOptsSoon();
  });
  canvas.addEventListener("pointercancel", () => { drag = null; });
  canvas.addEventListener("pointerleave", hideCursor);
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    if (spaceDims() === 3) { view3d?.zoom(Math.exp(-e.deltaY * 0.0015)); state.dirty = true; saveOptsSoon(); return; }
    const r = canvas.getBoundingClientRect(); renderer.zoom(Math.exp(-e.deltaY * 0.0015), e.clientX - r.left, e.clientY - r.top); viewCustom = true; viewLastChange = performance.now(); state.dirty = true; saveOptsSoon();
  }, { passive: false });
}
/** the free screen region right of the left column, with the same 12 px margin the panels keep from the viewport */
const MARGIN = 12;
function viewRegion(): [number, number, number, number] {
  const left = $("left").getBoundingClientRect();
  const r = canvas.getBoundingClientRect();
  return [left.right - r.left + MARGIN, MARGIN, r.width - MARGIN, r.height - MARGIN];
}
let viewCustom = false; // true after a pan / zoom; a fitted view is re-fitted on resize and never restored stale
const fitView = () => { renderer.fit(viewBox, viewRegion()); viewCustom = false; };
const refit = () => { if (spaceDims() === 3) view3d?.fit(); else fitView(); state.dirty = true; saveOptsSoon(); };
$("fit").onclick = refit;
// 3D view presets: look at the box's centre from the direction in data-look (xy / yz / xz face-on, xyz diagonal)
for (const b of document.querySelectorAll<HTMLButtonElement>("#viewBar [data-look]")) {
  b.onclick = () => {
    const dir = b.dataset.look!.split(",").map(Number) as [number, number, number];
    view3d?.look(dir);
    state.dirty = true; saveOptsSoon();
  };
}
/** after an orientation change: reflect it on the buttons and keep a fitted view fitted */
function orientationChanged(): void {
  $("flipx").classList.toggle("active", renderer.view.flipX);
  $("flipy").classList.toggle("active", renderer.view.flipY);
  if (!viewCustom) fitView();
  state.dirty = true; saveOptsSoon();
}
$("flipx").onclick = () => { renderer.view.flipX = !renderer.view.flipX; orientationChanged(); };
$("flipy").onclick = () => { renderer.view.flipY = !renderer.view.flipY; orientationChanged(); };
$("cw").onclick = () => { renderer.view.rot = ((renderer.view.rot + 1) % 4) as 0 | 1 | 2 | 3; orientationChanged(); };
$("ccw").onclick = () => { renderer.view.rot = ((renderer.view.rot + 3) % 4) as 0 | 1 | 2 | 3; orientationChanged(); };
/** panel shortcut key → the ✓ checkbox it toggles (the colorfield panel is absent in 3D, where the key is inert) */
const PANEL_KEYS: Partial<Record<string, "showScalar" | "showIso" | "showStream" | "showVec">> = { c: "showScalar", i: "showIso", s: "showStream", v: "showVec" };
// a picked <select> (bundle, space) keeps keyboard focus, and space would then reopen its menu instead of toggling
// play: once a choice is made, focus goes back to the page. Text inputs keep their keys (the guard below).
for (const sel of document.querySelectorAll<HTMLSelectElement>("select")) sel.addEventListener("change", () => sel.blur());
window.addEventListener("keydown", (e) => {
  const t = e.target as HTMLElement | null;
  if (t && (t.tagName === "TEXTAREA" || (t.tagName === "INPUT" && !/^(checkbox|radio|range|button)$/.test((t as HTMLInputElement).type)) || t.tagName === "SELECT")) return;
  if (e.key === "r" || e.key === "R") refit();
  // c / i / s / v toggle the gated panels (their underlined first letters = their columns in the mappings matrix);
  // plain keys only, so ⌘C / ⌘V / ⌘S keep their meaning
  if (!e.metaKey && !e.ctrlKey && !e.altKey && e.key in PANEL_KEYS) {
    const cb = ui[PANEL_KEYS[e.key]!];
    if (cb.closest<HTMLElement>(".panel")!.offsetParent !== null) { e.preventDefault(); cb.click(); }
  }
  if (e.key === "=" || e.key === "+") stepBoxZoom(-1); // zoom IN: a narrower domain
  if (e.key === "-" || e.key === "_") stepBoxZoom(1); // zoom OUT: a wider domain
  if (e.key === "0" && state.space && state.boxZoom[state.space]) { state.boxZoom = { ...state.boxZoom, [state.space]: 0 }; rebuildBundle(); refit(); status(""); }
  if (e.key === " ") {
    e.preventDefault();
    // the ▶s that can animate: those of enabled panels (a disabled panel's ▶ is inert). When none of them is on,
    // nothing is animating and space starts them all (and unpauses) instead of toggling a pause of nothing — so it
    // always does something, and what starts is unambiguous: with one panel disabled only the other one starts.
    // With both panels disabled it falls back to starting both.
    const pairs: [play: HTMLInputElement, panel: HTMLInputElement][] = [[ui.isoAnim, ui.showIso], [ui.anim, ui.showStream]];
    const live = pairs.filter(([, on]) => on.checked).map(([cb]) => cb);
    const starts = live.length ? live : [ui.isoAnim, ui.anim];
    if (!live.some((cb) => cb.checked)) {
      for (const cb of starts) if (!cb.checked) { cb.checked = true; cb.dispatchEvent(new Event("change")); }
      state.paused = false;
      status("");
    } else {
      state.paused = !state.paused;
      status(state.paused ? "animations paused (space to resume)" : "");
    }
    state.dirty = true;
  }
});

// control changes
for (const id of CHECKS) { ui[id].addEventListener("change", () => { state.dirty = true; saveOptsSoon(); }); }
// turning a ▶ on is a request to see it move: it lifts the pause (loads and space switches start paused)
for (const cb of [ui.isoAnim, ui.anim]) cb.addEventListener("change", () => { if (cb.checked && state.paused) { state.paused = false; status(""); } });
for (const id of VALUES) { ui[id].addEventListener("input", () => { state.dirty = true; }); ui[id].addEventListener("change", saveOptsSoon); }
for (const id of ["isoValue", "split"] as const) ui[id].addEventListener("input", () => { isoLastChange = performance.now(); });
for (const id of ["showScalar", "showIso", "showStream", "showVec"] as const) ui[id].addEventListener("change", () => { buildMetrics(); updateInfo(); });
ui.lines.addEventListener("change", () => { buildMetrics(); updateInfo(); });

/*******************************************************/
/* frame loop */

let lastT = performance.now();
/** the rendered frame awaiting its frame time (the interval to the next rAF includes the GPU stall) */
let pendingFrame: (Omit<FrameReport, "ms"> & { t0: number; jsMs: number; gpuMs?: number; awaitingGpu?: boolean }) | undefined;
let lastFrameKey = "", lastTier: Tier = "settled";
/** what a frame recomputes when it changes: grid, levels, slots, options (CPU compute has no dispatch counter to watch) */
const frameKey = () => [spaceDims(), state.revision, autoRes().resolution(tier()), ui.isoValue.value, ui.split.value, JSON.stringify(state.sel), ui.metric.value, ui.line.value, ui.isoExact.checked, ui.showIso.checked, viewBoxKey, cropCommitted.flat().join(",")].join("|");
/** what the resolution is spent on: failed steps are remembered per context */
const resCtx = () => [state.bundleFile, state.space, JSON.stringify(state.sel), ui.split.value, ui.metric.value, ui.line.value, ui.isoExact.checked, ui.isoOutline.checked, ui.showIso.checked, ui.showScalar.checked, ui.lines.value, modes.compute, modes.render].join("|");
/** debugging hook: per-frame GPU counters (`window.__tensatory.frames` = last 60 frames of { ms, dispatches, pipelines, recomputed }) */
const frameLog: { ms: number; dispatches: number; pipelines: number; recomputed: boolean }[] = [];
(window as unknown as { __tensatory: unknown }).__tensatory = { frames: frameLog, gpu: () => sampler.gpu, recolour: () => recolour, autores: () => autoRes(), state: () => state, yields: () => statusYields, view3d: () => view3d };

function frame(now: number): void {
  const frameMs = now - lastT; // the interval of the frame that just ended
  const dt = state.paused ? 0 : Math.min(0.1, frameMs / 1000);
  lastT = now;
  // a RENDER-ONLY frame carrying recolour batches is not the controller's business (its time is the recolouring,
  // not the draw — it once stepped the grid down for "slow draws"); a recomputed frame is always reported: a
  // multi-second remesh must step the settled tier down whatever else was in flight (its first slow sample only
  // triggers a remeasure, by which time the batches have drained — a step resets `stable` and stops them)
  // a settled-tier RECOMPUTE is timed by GPU completion (`onSubmittedWorkDone`), not by the rAF interval: submits
  // are asynchronous and the browser lets a backlog build for a frame or two before it blocks presenting, so a
  // 4 s remesh used to be measured as the 10 ms until the next rAF and the stall landed on a later, ignored frame
  if (pendingFrame?.awaitingGpu && pendingFrame.gpuMs === undefined) { /* still on the GPU: report when it lands */ }
  else if (pendingFrame) {
    const { t0, jsMs, gpuMs, awaitingGpu: _a, ...r } = pendingFrame; pendingFrame = undefined;
    if (!recolour?.busy || r.recomputed) {
      const ar = autoRes(), wasStable = ar.stable;
      ar.report({ ...r, jsMs, ms: Math.max(gpuMs ?? now - t0, jsMs) });
      // the ladder just decided to HOLD: the last render was refused registration for recolouring (not stable yet)
      // and, paused, nothing else would render again — so render once more now
      if (ar.stable && !wasStable) state.dirty = true;
    }
  }
  if (ui.anim.checked && !state.paused && num("lines") !== null && streamVector()) { state.animClock += state.dir.stream * dt; state.dirty = true; }
  if (spaceDims() === 3 && view3d?.flingTick(Math.min(0.1, frameMs / 1000))) state.dirty = true; // a flung orbit keeps turning (space does not pause it: it is not an animation of the data)
  if (isoAnimating() && slotScalar("iv")) {
    const cycle = Math.pow(10, 2 * +ui.isoRate.value!);
    ui.isoValue.value = String((((+ui.isoValue.value! + (state.dir.iso * dt) / cycle) % 1) + 1) % 1);
    state.dirty = true;
  }
  if (isoCache?.result.rough && !isoMoving() && !geometry?.busy) state.dirty = true; // settled: replace rough lines with exact ones
  controls.setEnabled(!animating());
  if (tier() !== lastTier) { lastTier = tier(); state.dirty = true; } // the levels settled (or started moving): switch resolution tier
  if (state.dirty && statusAwaitingPaint()) { statusYields++; requestAnimationFrame(frame); return; } // let "animations paused" paint before the heavy frame it triggers
  if (state.dirty) {
    Cache.frame++;
    const gpu = sampler.gpu, d0 = gpu?.dispatches ?? 0, p0 = gpu?.pipelinesBuilt ?? 0, t0 = performance.now(), usedTier = tier(), key = frameKey();
    try { recolour?.beginFrame(); render(); } catch (e) { showError(e); state.dirty = false; }
    const a = autoRes();
    const overCap = governMemory(a.capBytes);
    const bytes = memoryNow();
    if (pendingFrame?.awaitingGpu && pendingFrame.gpuMs === undefined) { /* a settled recompute is still being timed: this frame is not a sample */ }
    else {
      // "compiled": pipelines were built during this render — or recently: compiles are asynchronous, so the build
      // completes on one frame and the stall (Safari finishes the Metal compile at first submit, and presenting waits
      // for the queue) lands on the frames after it; none of those is a sample of the rung's own cost
      const built = gpu?.pipelinesBuilt ?? 0;
      if (built !== p0 || built !== lastBuilt || (gpu?.compiling ?? 0) > 0) compileTaint = COMPILE_TAINT_FRAMES;
      lastBuilt = built;
      const pf: NonNullable<typeof pendingFrame> = { t0, jsMs: performance.now() - t0, tier: usedTier, recomputed: key !== lastFrameKey || (gpu?.dispatches ?? 0) !== d0, compiled: compileTaint > 0, bytes, overCap, ctx: resCtx() };
      if (compileTaint > 0) compileTaint--;
      if (gpu && usedTier === "settled" && pf.recomputed) { pf.awaitingGpu = true; void gpu.device.queue.onSubmittedWorkDone().then(() => { pf.gpuMs = performance.now() - t0; }); }
      pendingFrame = pf;
    }
    frameLog.push({ ms: performance.now() - t0, dispatches: (gpu?.dispatches ?? 0) - d0, pipelines: (gpu?.pipelinesBuilt ?? 0) - p0, recomputed: key !== lastFrameKey || (gpu?.dispatches ?? 0) !== d0 }); if (frameLog.length > 60) frameLog.shift();
    lastFrameKey = key;
    $("memv").textContent = `${fmtMB(bytes.total)}/${fmtMB(a.capBytes)}${a.pin !== undefined ? ` · pinned ${a.pin}` : a.note ? ` · ${a.note}` : ""}`;
  }
  // the gear spins while a shader compiles (createComputePipelineAsync; the frame that needed it was not presented)
  const compiling = (sampler.gpu?.compiling ?? 0) > 0;
  if (compiling !== gearOn) { gearOn = compiling; $("gear").classList.toggle("on", compiling); }
  // progressive exact recolouring: a batch EVERY frame over the sets the last render registered, the image
  // refreshed every RECOLOUR_REFRESH frames (a re-render of the scene costs far more GPU than a batch)
  if (recolour?.pending) { const more = recolour.step(frameMs); if (more && ++recolourTick % RECOLOUR_REFRESH === 0) state.dirty = true; if (!more) state.dirty = true; }
  requestAnimationFrame(frame);
}
let gearOn = false, recolourTick = 0, statusYields = 0, lastBuilt = 0, compileTaint = 0;
/** rendered frames after a pipeline build (or while one is in flight) that the resolution controller treats as compiled */
const COMPILE_TAINT_FRAMES = 3;
const RECOLOUR_REFRESH = 4;

/*******************************************************/
/* boot */

(async () => {
  const params = new URLSearchParams(location.search);
  bootPhase("WebGPU device");
  await sampler.init("auto");
  console.log(`boot: compute backend ${sampler.label}`);
  if (sampler.gpu) sampler.gpu.onPipelineReady = () => { state.dirty = true; };
  sampler.check = params.get("check") === "1";
  try { Object.assign(modes, JSON.parse(localStorage.getItem("tensatory.modes") ?? "{}")); } catch { /* ignore */ }
  if (!localStorage.getItem("tensatory.modes") && sampler.gpu) { modes.compute = "gpu"; modes.render = "gpu"; } // default: fused when possible
  const wantCompute = params.get("compute") ?? params.get("backend"), wantRender = params.get("render");
  if (wantCompute === "cpu" || wantCompute === "gpu") modes.compute = wantCompute;
  if (wantRender === "canvas" || wantRender === "gpu") modes.render = wantRender;
  applyModes();
  memcapEl.value = params.get("memcap") ?? localStorage.getItem("tensatory.memcap") ?? "1024";
  applyMemcap();
  const pin2 = Number(params.get("res")), pin3 = Number(params.get("res3"));
  if (pin2 > 1) autoRes2.pin = pin2;
  if (pin3 > 1) autoRes3.pin = pin3;
  bootPhase("bundles/index.json");
  try {
    bundleList = (await (await fetch("bundles/index.json", { cache: "no-cache" })).json()) as typeof bundleList;
  } catch (e) { console.error(e); }
  if (!bundleList.length) { bootPhase(undefined); status("no bundles found in bundles/index.json"); requestAnimationFrame(frame); return; }
  pickSel.replaceChildren(...bundleList.map((b) => Object.assign(document.createElement("option"), { value: b.file, textContent: b.name ?? b.file, title: b.summary ?? "" })));
  const want = params.get("bundle");
  const file = want && bundleList.some((b) => b.file === want) ? want : bundleList[0]!.file;
  pickSel.value = file;
  syncPickers();
  bootPhase(`bundle ${file}`);
  await loadBundle(file, params.get("space"), params.get("member"));
  bootPhase(undefined);
  // UI overrides from the URL, e.g. &showIso=0&split=3&iv=loss&sg=lossGrad
  for (const [k, v] of params) {
    if ((CHECKS as readonly string[]).includes(k)) ui[k as CheckId].checked = v !== "0";
    else if ((VALUES as readonly string[]).includes(k)) ui[k as ValueId].value = v === "null" ? null : v;
    else if ((SLOTS as readonly string[]).includes(k)) { const id = v === "none" ? NONE : isUsable(v) ? v : undefined; if (id !== undefined) setSel({ [k]: id }, true); }
  }
  syncTicks(); syncIsoRate(); readCrop(); buildMetrics(); updateInfo(); fitLeftColumn();
  state.dirty = true;
  requestAnimationFrame(frame);
})();
