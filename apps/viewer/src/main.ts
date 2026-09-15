// Tensatory 2D viewer: wiring between the bundle runtime (@tensatory/core),
// the widgets (widgets.ts / metrics.ts) and the canvas renderer (render2d.ts).

import {
  Box,
  Bundle,
  Codomain,
  DenseGrid,
  DenseVectorFieldData,
  formatReal,
  SymbolicScalarFieldData,
  SymbolicVectorFieldData,
  TensatoryError,
  boxBlur,
  contourField,
  integrateFromSeeds,
  isoContours,
  streamlineSeeds,
  taubinSmooth,
  type ContourResult,
  type Polyline,
  type ScalarFieldData,
  type Streamline,
  type VectorFieldData,
} from "@tensatory/core";
import { MAPS, cmap, cssGradient, lut } from "./colormap";
import { installLogCapture, showError, status } from "./log";
import { MetricsTable, NONE, type MetricsRow, type Sel } from "./metrics";
import { Renderer2D, type LineLayer, type Scene } from "./render2d";
import { Sampler, type Values } from "./sampler";
import { GpuGeometry } from "./gpuGeometry";
import {
  installCollapsiblePanels,
  installTicks,
  fmtNum,
  installTooltips,
  makeDiscreteSlider,
  makeSlider,
  syncTicks,
  wheelStepper,
  type ValueControl,
} from "./widgets";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

/*******************************************************/
/* widgets */

installLogCapture();
installTooltips();
installTicks();
for (const el of document.querySelectorAll<HTMLElement>(".sl")) makeSlider(el);
for (const el of document.querySelectorAll<HTMLElement>(".ds")) makeDiscreteSlider(el);

const CHECKS = ["showPoints", "showBox", "showScalar", "smooth", "showIso", "isoAnim", "showStream", "anim"] as const;
const VALUES = ["res", "isoRate", "isoValue", "split", "isoAlpha", "metric", "line", "lines", "slen", "sAlpha", "tail", "ssplit"] as const;
type CheckId = (typeof CHECKS)[number];
type ValueId = (typeof VALUES)[number];
const ui = {
  ...(Object.fromEntries(CHECKS.map((id) => [id, $<HTMLInputElement>(id)])) as Record<CheckId, HTMLInputElement>),
  ...(Object.fromEntries(VALUES.map((id) => [id, $<HTMLElement>(id) as ValueControl])) as Record<ValueId, ValueControl>),
};
const num = (id: ValueId): number | null => { const v = ui[id].value; return v === null ? null : +v; };
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

const SLOTS = ["c", "iv", "ic", "sg", "sc"] as const;
type Slot = (typeof SLOTS)[number];
const SLOT_HTML: Record<Slot, string> = { c: "C", iv: "I<sub>V</sub>", ic: "I<sub>C</sub>", sg: "S<sub>∇</sub>", sc: "S<sub>C</sub>" };
const SLOT_TIP: Record<Slot, string> = {
  c: "colorfield: the field painted as a colormapped raster",
  iv: "isoline value: the field whose level sets are drawn",
  ic: "isoline colour",
  sg: "streamline direction: a vector field, or the gradient of a scalar field",
  sc: "streamline colour",
};

interface State {
  bundle: Bundle | undefined;
  bundleFile: string; // for options storage
  sel: Sel; // shown (may be a hover preview)
  lockedSel: Sel;
  /** colormap index per scalar use id */
  maps: Record<string, number>;
  dirty: boolean;
  paused: boolean;
  animClock: number;
  /** animation directions (+1 forward, -1 backward); shift-click a ▶ to reverse. For streamlines the
   *  direction is the flow direction itself (-1 = against the field, i.e. descent for a gradient) */
  dir: { iso: 1 | -1; stream: 1 | -1 };
}
const emptySel = (): Sel => Object.fromEntries(SLOTS.map((k) => [k, NONE]));
const state: State = { bundle: undefined, bundleFile: "", sel: emptySel(), lockedSel: emptySel(), maps: {}, dirty: true, paused: false, animClock: 0, dir: { iso: 1, stream: -1 } };
const canvas = $<HTMLCanvasElement>("gl");
const renderer = new Renderer2D(canvas);
const sampler = new Sampler(() => { state.dirty = true; });
let geometry: GpuGeometry | undefined; // set after the sampler finds a GPU
let usable = { scalars: [] as string[], vectors: [] as string[] };

/*******************************************************/
/* field uses: a slot resolves a field id to the field itself, the gradient of a
   scalar (for vector slots) or the norm of a vector (for scalar slots) */

interface ScalarUse { id: string; name: string; codomain: Codomain; data: ScalarFieldData }
interface VectorUse { id: string; name: string; data: VectorFieldData }
const useCache = new Map<string, ScalarUse | VectorUse>();
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
    u = { id: key, name: `|${v.name}|`, codomain: new Codomain("norm"), data: new SymbolicScalarFieldData({ k: "norm", v: { k: "argv", name: "v" } }, 2, { scalars: {}, vectors: { v: v.data } }) };
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
    u = { id: key, name: `∇${paren(f.name)}`, data: new SymbolicVectorFieldData({ k: "grad", s: { k: "arg", name: "f" } }, 2, { scalars: { f: f.data }, vectors: {} }) };
    useCache.set(key, u);
  }
  return u;
}

const slotScalar = (k: Slot): ScalarUse | undefined => useScalar(state.sel[k]);
const streamsOn = () => ui.showStream.checked && num("lines") !== null;
const streamVector = (): VectorUse | undefined => (streamsOn() ? useVector(state.sel.sg) : undefined);

/*******************************************************/
/* per-use value ranges (for colormaps and the value slider) */

const rangeCache = new Map<string, [number, number]>();
function rangeOf(f: ScalarUse): [number, number] {
  let r = rangeCache.get(f.id);
  if (r) return r;
  const st = f.data.stats(), cd = f.codomain;
  let lo = Math.max(st.min, cd.min), hi = Math.min(st.max, cd.max);
  if (cd.log && !(lo > 0)) {
    const g = f.data.samplePoints ?? new DenseGrid([64, 64], f.data.box);
    let m = Infinity;
    for (const v of f.data.sampleOn(g)) if (v > 0 && v < m) m = v;
    lo = Number.isFinite(m) ? m : 1e-9;
  }
  if (!(hi > lo)) hi = lo + 1e-9;
  rangeCache.set(f.id, (r = [lo, hi]));
  return r;
}
const paramOf = (f: ScalarUse) => { const [lo, hi] = rangeOf(f); const cd = f.codomain; return (v: number) => (Number.isNaN(v) ? NaN : cd.toParam(Math.min(hi, Math.max(lo, v)), lo, hi)); };
const mapOf = (f: ScalarUse) => cmap(state.maps[f.id] ?? 0);

/*******************************************************/
/* the view box and sampling grid */

function selectedUses(): { scalars: ScalarUse[]; vector: VectorUse | undefined } {
  const uses = new Map<string, ScalarUse>();
  const add = (u: ScalarUse | undefined) => { if (u) uses.set(u.id, u); };
  if (ui.showScalar.checked) add(slotScalar("c"));
  if (ui.showIso.checked) { add(slotScalar("iv")); add(slotScalar("ic")); }
  const vector = streamVector();
  if (vector) add(slotScalar("sc"));
  return { scalars: [...uses.values()], vector };
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
  const { scalars, vector } = selectedUses();
  let boxes = [...scalars.map((f) => f.data.box), ...(vector ? [vector.data.box] : [])];
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
  const res = num("res");
  if (res === null) {
    // native: when every selected sampled field shares one grid filling the view box
    const grids = selectedUses().scalars.map((f) => f.data.samplePoints).filter((g): g is DenseGrid => !!g);
    if (grids.length && grids.every((g) => g.equals(grids[0]!, 1e-12)) && grids[0]!.box.equals(box, 1e-12)) return grids[0]!;
    return squareGrid(box, 128);
  }
  return squareGrid(box, res);
}

/** field values on the grid (NaN outside the field's own box); undefined while the GPU is still computing */
function sampleUse(u: ScalarUse, grid: DenseGrid): Values | undefined {
  return sampler.request(`${u.id}|${grid.size.join("x")}|${grid.box.intervals.flat().join(",")}`, u.data, grid);
}

/*******************************************************/
/* isolines */

/** `split` levels evenly spaced and centred on `value` (codomain parameter space), wrapping past min/max */
function isoLevelParams(): number[] {
  const c = +ui.isoValue.value!;
  const split = num("split");
  if (split === null) return [c];
  const out: number[] = [];
  for (let k = 0; k < split; k++) { let l = c - 0.5 + (k + 0.5) / split; l = ((l % 1) + 1) % 1; out.push(l); }
  return out;
}

interface IsoResult { lines: Polyline[]; values?: (Float64Array | undefined)[]; info: string; rough: boolean }
let isoCache: { key: string; result: IsoResult } | undefined;
/** while the level is moving (animation, slider drag) contours are plain marching squares; exact projection follows once it settles */
let isoLastChange = -1e9;
const ISO_SETTLE_MS = 200;
const isoMoving = () => (ui.isoAnim.checked && !state.paused) || performance.now() - isoLastChange < ISO_SETTLE_MS;
function isolines(grid: DenseGrid): IsoResult | undefined {
  const f = slotScalar("iv");
  if (!f || !ui.showIso.checked) return undefined;
  const metric = num("metric"), line = num("line") ?? 0;
  const tol = 0.25 * renderer.worldPerPixel;
  const ic = slotScalar("ic");
  const exactWanted = f.data.kind === "symbolic" && metric === null;
  let rough = isoMoving() && exactWanted;
  const gridKey = `${grid.size.join("x")}|${viewBoxKey}`;
  const key = [f.id, gridKey, metric, line, ui.isoValue.value, num("split"), tol.toExponential(2), ic?.id ?? "", rough].join("|");
  if (isoCache?.key === key) return isoCache.result;
  const raw = sampleUse(f, grid);
  if (!raw) return undefined; // still sampling
  const t0 = performance.now();
  const values = metric === null ? raw : boxBlur(grid, raw, metric);
  const [lo, hi] = rangeOf(f);
  const lines: Polyline[] = [];
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
  }
  for (const l of lines) vertices += l.length / 2;
  let colours: (Float64Array | undefined)[] | undefined;
  if (ic) {
    const toParam = paramOf(ic);
    colours = lines.map((l) => {
      const out = new Float64Array(l.length / 2);
      for (let i = 0; i < out.length; i++) { const v = ic.data.value([l[2 * i]!, l[2 * i + 1]!]); out[i] = v === undefined ? 0 : toParam(v); }
      return out;
    });
  }
  const ms = (performance.now() - t0).toFixed(1);
  const info = method === "exact" ? `exact${gpuResults ? " (GPU)" : ""}: ${vertices} vertices, max |f − c| = ${maxRes.toExponential(1)}, ${ms} ms` : `marching squares on ${grid.size.join("×")}: ${vertices} vertices, ${ms} ms`;
  const result: IsoResult = { lines, values: colours, info, rough };
  // a rough result standing in for pending GPU lines is keyed as rough so the exact one replaces it when it lands
  isoCache = { key: [f.id, gridKey, metric, line, ui.isoValue.value, num("split"), tol.toExponential(2), ic?.id ?? "", rough].join("|"), result };
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

let stream: StreamSet | undefined;
function streamlines(grid: DenseGrid): StreamSet | undefined {
  const v = streamVector();
  const count = num("lines");
  if (!v || count === null) { stream = undefined; return undefined; }
  const maxSteps = num("slen")!;
  const sign = state.dir.stream;
  const cell = Math.min(grid.spacing[0]!, grid.spacing[1]!) || 1e-3;
  const step = 0.5 * cell;
  const field = integrableVector(v, grid);
  if (!field) { stream = undefined; return undefined; } // still sampling
  const box = field.box.intersect(grid.box) ?? field.box;
  const key = [v.id, count, maxSteps, sign, step.toExponential(4), box.intervals.flat().join(",")].join("|");
  let set = STREAM_CACHE.get(key);
  if (!set) {
    const seeds = streamlineSeeds(box, count, 12345);
    let lines: Streamline[] | undefined;
    if (geometry) {
      lines = geometry.streamlines(key, field, seeds, { maxSteps, step, sign, box });
      if (!lines) return stream; // still integrating: keep showing the previous set
    } else {
      const t0 = performance.now();
      lines = integrateFromSeeds(field, seeds, { maxSteps, step, sign, box });
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
        for (let i = 0; i < out.length; i++) { const val = s.data.value([l.points[2 * i]!, l.points[2 * i + 1]!]); out[i] = val === undefined ? 0 : toParam(val); }
        return out;
      });
    }
  }
  stream = set;
  return set;
}

/*******************************************************/
/* render */

function render(): void {
  state.dirty = false;
  if (!state.bundle) { renderer.render({ box: Box.unit(2), crop: [1, 1], showBox: false, lines: [], pointSets: [] }); return; }
  const box = currentViewBox();
  const grid = currentGrid(box);
  const scene: Scene = {
    box, crop: [1, 1], showBox: ui.showBox.checked, lines: [],
    pointSets: ui.showPoints.checked ? [...state.bundle.pointSets.values()].filter((ps) => ps.domain.numDims === 2) : [],
  };
  const c = slotScalar("c");
  const cValues = ui.showScalar.checked && c ? sampleUse(c, grid) : undefined;
  if (c && cValues) {
    const values = cValues;
    scene.raster = { key: `${c.id}|${grid.size.join("x")}|${viewBoxKey}|${state.maps[c.id] ?? 0}`, grid, values, toParam: paramOf(c), lut: lut(mapOf(c)), smooth: ui.smooth.checked };
  }
  const iso = isolines(grid);
  const isoField = slotScalar("iv");
  if (iso) {
    const alpha = num("isoAlpha") ?? 1;
    const layer: LineLayer = { lines: iso.lines, color: [0.92, 0.92, 0.92], width: 2, alpha };
    const ic = slotScalar("ic");
    if (iso.values && ic) { layer.values = iso.values; layer.cmap = mapOf(ic); }
    scene.lines.push(layer);
  }
  const st = streamlines(grid);
  if (st) {
    const layer: LineLayer = { lines: st.lines.map((l) => l.points), color: [1, 1, 1], width: 1.5, alpha: num("sAlpha") ?? 1 };
    const sc = slotScalar("sc");
    if (st.colours && sc) { layer.values = st.colours; layer.cmap = mapOf(sc); }
    // particles are always drawn at the current phase; ▶ only advances the clock (as in the 3D prototype)
    layer.particles = { lengths: st.lines.map((l) => l.length), phases: st.lines.map((l) => l.phase), step: st.step, tail: (num("tail") ?? 5) * st.cell, split: num("ssplit") ?? 1, travel: state.animClock * 10 * st.cell };
    scene.lines.push(layer);
  }
  renderer.render(scene);
  updateIsoNotches();

  // labels
  $("resv").textContent = num("res") === null ? `${grid.size.join("×")}` : String(num("res"));
  $("isoValuev").textContent = isoField ? isoField.codomain.format(isoField.codomain.fromParam(+ui.isoValue.value!, ...rangeOf(isoField))) : "—";
  $("splitv").textContent = ui.split.value ?? "—";
  $("isoAlphav").textContent = ui.isoAlpha.value === null ? "—" : (+ui.isoAlpha.value).toFixed(2);
  $("metricv").textContent = ui.metric.value ?? "—";
  $("linev").textContent = ui.line.value ?? "—";
  // isoline diagnostics ("exact: N vertices, max |f − c| …" / "marching squares on …"); the #isoInfo row is commented out in index.html
  // $("isoInfo").textContent = iso?.info ?? "";
  $("linesv").textContent = ui.lines.value === null ? "—" : fmtNum(+ui.lines.value);
  $("slenv").textContent = ui.slen.value ?? "";
  $("sAlphav").textContent = ui.sAlpha.value === null ? "—" : (+ui.sAlpha.value).toFixed(2);
  $("tailv").textContent = ui.tail.value ?? "";
  $("ssplitv").textContent = ui.ssplit.value ?? "—";
}

/*******************************************************/
/* legend (#info): one bar per distinct coloured field, listing the slots that use it */

const fmt3 = (v: number) => formatReal(v, 3);
function centerPoint(): number[] | undefined {
  const ps = state.bundle && [...state.bundle.pointSets.values()].find((p) => p.points.length === 1 && p.domain.numDims === 2);
  return ps?.points[0];
}
/** the colour slots currently in use, in display order */
function colourSlots(): [Slot, ScalarUse][] {
  const out: [Slot, ScalarUse][] = [];
  const c = slotScalar("c"); if (ui.showScalar.checked && c) out.push(["c", c]);
  const ic = slotScalar("ic"); if (ui.showIso.checked && ic) out.push(["ic", ic]);
  const sc = slotScalar("sc"); if (streamVector() && sc) out.push(["sc", sc]);
  return out;
}
const legendUses = new Map<string, ScalarUse>();
/** slots that shape the picture without colouring it: their fields get an all-black bar (min, max, detent, notches, pip) */
function shapeSlots(): [Slot, ScalarUse][] {
  const out: [Slot, ScalarUse][] = [];
  const iv = slotScalar("iv"); if (ui.showIso.checked && iv) out.push(["iv", iv]);
  const sg = state.sel.sg; if (streamsOn() && sg && usable.scalars.includes(sg)) { const u = useScalar(sg); if (u) out.push(["sg", u]); }
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
    const bg = coloured ? cssGradient(map) : "#000";
    const title = coloured ? `${MAPS[map % MAPS.length]}${cd.log ? `, log${cd.log}` : ""} — click to cycle colormap` : `not used for colour${cd.log ? ` (log${cd.log})` : ""}`;
    const order: Slot[] = ["c", "iv", "ic", "sg", "sc"];
    const slotHtml = order.filter((k) => slots.includes(k)).map((k) => SLOT_HTML[k]).join(" ");
    rows.push(`<div class="lrow" data-use="${f.id}"><span class="lname" title="${f.name}"><span class="slot">${slotHtml}</span>${f.name}</span><span class="lval">${cd.format(l)}</span><span class="bar${coloured ? "" : " plain"}" data-use="${f.id}" style="background:${bg}${coloured ? ";cursor:pointer" : ""}" title="${title}"><span class="notches"></span>${detent}<span class="pip" data-pip="${f.id}"></span></span><span class="lval">${cd.format(r)}</span></div>`);
  }
  const info = $("info");
  info.innerHTML = rows.join("");
  info.style.display = rows.length ? "" : "none";
  for (const el of info.querySelectorAll<HTMLElement>(".bar:not(.plain)")) el.onclick = () => { const id = el.dataset.use!; state.maps[id] = ((state.maps[id] ?? 0) + 1) % MAPS.length; updateInfo(); state.dirty = true; saveOptsSoon(); };
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
  const lines: string[] = [`<b>${esc(b.defaultManifold.name)}:</b> ${fmt3(x)} ${fmt3(y)}`];
  const vf = streamVector();
  if (vf) { const g = vf.data.value([x, y]); if (g) lines.push(`<b>${esc(vf.name)}:</b> ${g.map(fmt3).join(" ")}`); }
  const seen = new Set<string>();
  const uses = [...colourSlots(), ...shapeSlots()].sort((a, b) => ["c", "iv", "ic", "sg", "sc"].indexOf(a[0]) - ["c", "iv", "ic", "sg", "sc"].indexOf(b[0])).map(([, u]) => u);
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
    { key: "c", label: "C", tip: SLOT_TIP.c, type: "scalar", visible: () => ui.showScalar.checked, toggle: () => ui.showScalar.click() },
    { key: "iv", label: "I", sub: "V", tip: SLOT_TIP.iv, type: "scalar", visible: () => ui.showIso.checked, toggle: () => ui.showIso.click() },
    { key: "ic", label: "I", sub: "C", tip: SLOT_TIP.ic, type: "scalar", visible: () => ui.showIso.checked, toggle: () => ui.showIso.click() },
    { key: "sg", label: "S", sub: "∇", tip: SLOT_TIP.sg, type: "vector", visible: streamsOn, toggle: () => ui.showStream.click() },
    { key: "sc", label: "S", sub: "C", tip: SLOT_TIP.sc, type: "scalar", visible: streamsOn, toggle: () => ui.showStream.click() },
  ],
  sel: () => state.sel,
  lockedSel: () => state.lockedSel,
  setSel,
});
function setSel(partial: Sel, lock: boolean): void {
  let changed = false;
  for (const [k, id] of Object.entries(partial)) {
    if (lock) { state.lockedSel[k] = id; saveOptsSoon(); }
    if (state.sel[k] === id) continue;
    state.sel[k] = id; changed = true;
  }
  if (changed) {
    try { updateInfo(); } catch (e) { showError(e); }
    state.dirty = true;
  }
  metrics.refresh();
}
function matrixRows(): MetricsRow[] {
  const b = state.bundle!;
  return [
    ...usable.scalars.map((id): MetricsRow => ({ id, name: b.scalarField(id).name, kind: "scalar" })),
    ...usable.vectors.map((id): MetricsRow => ({ id, name: b.vectorField(id).name, kind: "vector" })),
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
window.addEventListener("resize", () => { fitLeftColumn(); if (!viewCustom && state.bundle) fitView(); state.dirty = true; });
installCollapsiblePanels("tensatory.collapsed", fitLeftColumn);

/*******************************************************/
/* options persistence (per bundle) */

let loadingOpts = false, saveTimer: ReturnType<typeof setTimeout> | undefined;
const optsKey = () => (state.bundleFile ? `tensatory.opts.${state.bundleFile}` : null);
function saveOpts(): void {
  const key = optsKey(); if (!key || loadingOpts) return;
  const o = {
    ui: Object.fromEntries([...CHECKS.map((id) => [id, ui[id].checked]), ...VALUES.map((id) => [id, ui[id].value])]),
    sel: state.lockedSel, maps: state.maps, view: viewCustom ? renderer.view : { flipX: renderer.view.flipX, flipY: renderer.view.flipY, rot: renderer.view.rot }, dir: state.dir,
  };
  localStorage.setItem(key, JSON.stringify(o));
}
const saveOptsSoon = () => { clearTimeout(saveTimer); saveTimer = setTimeout(saveOpts, 150); };
const isUsable = (id: string) => usable.scalars.includes(id) || usable.vectors.includes(id);
function loadOpts(): boolean {
  const key = optsKey(); const raw = key && localStorage.getItem(key); if (!raw) return false;
  let o: { ui?: Record<string, unknown>; sel?: Sel; maps?: Record<string, number>; view?: Partial<typeof renderer.view>; dir?: State["dir"] };
  try { o = JSON.parse(raw); } catch { return false; }
  loadingOpts = true;
  try {
    for (const [id, v] of Object.entries(o.ui ?? {})) {
      if ((CHECKS as readonly string[]).includes(id)) ui[id as CheckId].checked = Boolean(v);
      else if ((VALUES as readonly string[]).includes(id)) ui[id as ValueId].value = v as string | null;
    }
    syncTicks(); syncIsoRate();
    if (o.maps) state.maps = { ...o.maps };
    if (o.sel) for (const k of SLOTS) { const id = o.sel[k]; if (id === null || (id && isUsable(id))) state.sel[k] = state.lockedSel[k] = id ?? NONE; }
    if (o.view) { renderer.view = { ...renderer.view, ...o.view }; viewCustom = o.view.scale !== undefined; }
    if (o.dir) state.dir = { iso: o.dir.iso === -1 ? -1 : 1, stream: o.dir.stream === -1 ? -1 : 1 };
    syncPlayGlyphs();
  } finally { loadingOpts = false; }
  return true;
}

/*******************************************************/
/* bundle loading */

function setBundle(bundle: Bundle, file: string): void {
  state.bundle = bundle; state.bundleFile = file;
  rangeCache.clear(); sampler.clear(); geometry?.clear(); useCache.clear(); STREAM_CACHE.clear(); SAMPLED_VECTORS.clear(); isoCache = undefined; viewBoxKey = ""; state.maps = {};
  const errors = bundle.buildAll();
  usable = {
    scalars: bundle.scalarFieldIds.filter((id) => !errors.has(id) && bundle.scalarField(id).domain.numDims === 2),
    vectors: bundle.vectorFieldIds.filter((id) => !errors.has(id) && bundle.vectorField(id).domain.numDims === 2),
  };
  $("pickAbout").textContent = [bundle.name, bundle.spec.description ?? ""].filter(Boolean).join(" — ");
  $("pickSpace").textContent = [...bundle.manifolds.values()].map((m) => `${m.name}: ${m.numDims}D (${m.dimNames.join(", ")})`).join("\n") + `\n${usable.scalars.length} scalar, ${usable.vectors.length} vector fields, ${bundle.pointSets.size} point sets`;
  $("pickErrRow").style.display = errors.size ? "" : "none";
  $("pickErr").textContent = [...errors].map(([id, e]) => `${id}: ${e.message}`).join("\n");

  // defaults: colorfield and isoline value on the first scalar field; colour slots none (white lines;
  // a colour slot equal to C would make lines vanish into the raster); streamline direction from the
  // exact gradient when the bundle has one, otherwise the symbolic gradient of the field itself
  const first = usable.scalars[0] ?? NONE;
  const exact = first ? bundle.scalarField(first).spec.exactGradient : undefined;
  const sel: Sel = { c: first, iv: first, ic: NONE, sg: exact && usable.vectors.includes(exact) ? exact : first, sc: NONE };
  state.sel = { ...sel }; state.lockedSel = { ...sel };
  state.dir = { iso: 1, stream: -1 }; syncPlayGlyphs();
  viewCustom = false;
  renderer.view = { ...renderer.view, flipX: false, flipY: false, rot: 0 };
  currentViewBox(); // establishes the view box (and a default fit) before a saved view may override it
  loadOpts();
  if (!viewCustom) fitView();
  $("streamBox").style.display = usable.scalars.length + usable.vectors.length ? "" : "none";
  buildMetrics(); updateInfo();
  $("flipx").classList.toggle("active", renderer.view.flipX);
  $("flipy").classList.toggle("active", renderer.view.flipY);
  status("");
  state.dirty = true;
}

async function loadBundle(file: string): Promise<void> {
  status(`loading ${file}…`);
  try {
    const res = await fetch(`bundles/${file}`, { cache: "no-cache" });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText} for bundles/${file}`);
    setBundle(Bundle.parse(await res.json()), file);
    history.replaceState(null, "", `?bundle=${encodeURIComponent(file)}`);
  } catch (e) {
    console.error(e);
    status(e instanceof TensatoryError || e instanceof Error ? e.message : String(e));
  }
}

let bundleList: { file: string; name?: string }[] = [];
const pickSel = $<HTMLSelectElement>("pickBundle");
function chooseBundle(i: number): void {
  if (!bundleList.length) return;
  const b = bundleList[((i % bundleList.length) + bundleList.length) % bundleList.length]!;
  pickSel.value = b.file;
  void loadBundle(b.file);
}
pickSel.onchange = () => void loadBundle(pickSel.value);
{
  let over = false;
  pickSel.addEventListener("pointerenter", () => (over = true)); pickSel.addEventListener("pointerleave", () => (over = false));
  // same one-step-per-gesture wheel handling as the discrete sliders
  const wheel = wheelStepper((dir) => chooseBundle(pickSel.selectedIndex + dir));
  pickSel.addEventListener("wheel", (e) => { if (e.shiftKey) return; wheel(e); }, { passive: false });
  window.addEventListener("keydown", (e) => { if (!over || e.shiftKey) return; if (e.key === "ArrowDown") { e.preventDefault(); chooseBundle(pickSel.selectedIndex + 1); } else if (e.key === "ArrowUp") { e.preventDefault(); chooseBundle(pickSel.selectedIndex - 1); } });
}
$("uploadBtn").onclick = () => $<HTMLInputElement>("pickFile").click();
$<HTMLInputElement>("pickFile").addEventListener("change", async (ev) => {
  const file = (ev.target as HTMLInputElement).files?.[0]; if (!file) return;
  try { setBundle(Bundle.parse(JSON.parse(await file.text())), `local:${file.name}`); $("pickAbout").textContent = `${file.name} (local) — ${$("pickAbout").textContent}`; }
  catch (e) { console.error(e); status(e instanceof Error ? e.message : String(e)); }
});

/*******************************************************/
/* interaction */

{
  let drag: { x: number; y: number } | null = null;
  canvas.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY }; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener("pointermove", (e) => {
    if (drag) { renderer.pan(e.clientX - drag.x, e.clientY - drag.y); drag = { x: e.clientX, y: e.clientY }; viewCustom = true; state.dirty = true; return; }
    const r = canvas.getBoundingClientRect();
    const [x, y] = renderer.toWorld(e.clientX - r.left, e.clientY - r.top);
    showCursor(x, y);
  });
  canvas.addEventListener("pointerup", () => { drag = null; saveOptsSoon(); });
  canvas.addEventListener("pointerleave", hideCursor);
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("wheel", (e) => { e.preventDefault(); const r = canvas.getBoundingClientRect(); renderer.zoom(Math.exp(-e.deltaY * 0.0015), e.clientX - r.left, e.clientY - r.top); viewCustom = true; state.dirty = true; saveOptsSoon(); }, { passive: false });
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
const refit = () => { fitView(); state.dirty = true; saveOptsSoon(); };
$("fit").onclick = refit;
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
window.addEventListener("keydown", (e) => {
  if (e.target && /^(INPUT|SELECT|TEXTAREA)$/.test((e.target as HTMLElement).tagName)) return;
  if (e.key === "r" || e.key === "R") refit();
  if (e.key === " ") {
    e.preventDefault();
    if (!ui.isoAnim.checked && !ui.anim.checked) {
      // nothing is animating: space starts both (and unpauses) instead of toggling a pause of nothing
      for (const cb of [ui.isoAnim, ui.anim]) { cb.checked = true; cb.dispatchEvent(new Event("change")); }
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
for (const id of VALUES) { ui[id].addEventListener("input", () => { state.dirty = true; }); ui[id].addEventListener("change", saveOptsSoon); }
for (const id of ["isoValue", "split"] as const) ui[id].addEventListener("input", () => { isoLastChange = performance.now(); });
for (const id of ["showScalar", "showIso", "showStream"] as const) ui[id].addEventListener("change", () => { buildMetrics(); updateInfo(); });
ui.lines.addEventListener("change", () => { buildMetrics(); updateInfo(); });

/*******************************************************/
/* frame loop */

let lastT = performance.now();
function frame(now: number): void {
  const dt = state.paused ? 0 : Math.min(0.1, (now - lastT) / 1000);
  lastT = now;
  if (ui.anim.checked && !state.paused && stream) { state.animClock += dt; state.dirty = true; }
  if (ui.isoAnim.checked && !state.paused && slotScalar("iv")) {
    const cycle = Math.pow(10, 2 * +ui.isoRate.value!);
    ui.isoValue.value = String((((+ui.isoValue.value! + (state.dir.iso * dt) / cycle) % 1) + 1) % 1);
    state.dirty = true;
  }
  if (isoCache?.result.rough && !isoMoving() && !geometry?.busy) state.dirty = true; // settled: replace rough lines with exact ones
  if (state.dirty) {
    try { render(); } catch (e) { showError(e); state.dirty = false; }
  }
  requestAnimationFrame(frame);
}

/*******************************************************/
/* boot */

(async () => {
  const params = new URLSearchParams(location.search);
  const prefer = params.get("backend");
  await sampler.init(prefer === "cpu" || prefer === "gpu" ? prefer : "auto");
  sampler.check = params.get("check") === "1";
  if (sampler.gpu && params.get("geometry") !== "cpu") geometry = new GpuGeometry(sampler.gpu, () => { state.dirty = true; });
  $("pickCompute").textContent = sampler.label + (sampler.check ? " — agreement check on (see L)" : "");
  try {
    bundleList = (await (await fetch("bundles/index.json", { cache: "no-cache" })).json()) as typeof bundleList;
  } catch (e) { console.error(e); }
  if (!bundleList.length) { status("no bundles found in bundles/index.json"); requestAnimationFrame(frame); return; }
  pickSel.replaceChildren(...bundleList.map((b) => Object.assign(document.createElement("option"), { value: b.file, textContent: b.name ?? b.file })));
  const want = params.get("bundle");
  const file = want && bundleList.some((b) => b.file === want) ? want : bundleList[0]!.file;
  pickSel.value = file;
  await loadBundle(file);
  // UI overrides from the URL, e.g. &showIso=0&split=3&iv=loss&sg=lossGrad
  for (const [k, v] of params) {
    if ((CHECKS as readonly string[]).includes(k)) ui[k as CheckId].checked = v !== "0";
    else if ((VALUES as readonly string[]).includes(k)) ui[k as ValueId].value = v === "null" ? null : v;
    else if ((SLOTS as readonly string[]).includes(k)) { const id = v === "none" ? NONE : isUsable(v) ? v : undefined; if (id !== undefined) setSel({ [k]: id }, true); }
  }
  syncTicks(); syncIsoRate(); buildMetrics(); updateInfo(); fitLeftColumn();
  state.dirty = true;
  requestAnimationFrame(frame);
})();
