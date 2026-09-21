// The curves panel: one row per curve of the current space. The label (the curve's name) toggles drawing; the
// interval slider restricts the drawn parameter range — no interval = the whole curve, from t0 to t1 — with the
// same gestures as the 3D crop ranges (drag out a range, click a handle to open that end, Backspace to clear);
// the readout shows the shown range in the curve's parameter (`param`: name / unit / codomain formatting); the ⓘ
// exposes summary / details like every other bundle citizen. Everything is a per-bundle option (`curves` in the
// saved options): { on, lo, hi } by curve id.

import { Codomain, formatReal, type Curve } from "@tensatory/core";
import { bindInfoIcon } from "./info";
import { makeIntervalSlider, type IntervalEl } from "./interval";
import { installTooltips } from "./widgets";

export interface CurveOpts { on?: boolean; lo?: number | null; hi?: number | null }

/** what the renderers draw for a curve: its polyline over the shown range (flat [n, D]), sample markers (sampled data, with labels), the end point */
export interface CurveDrawable { id: string; points: Float64Array; dimCount: number; markers: { p: number[]; label?: string | undefined }[]; head: number[] | undefined }

/** the shown parameter range of a curve under `o` */
export const curveRange = (c: Curve, o: CurveOpts | undefined): [number, number] => [o?.lo ?? c.data.t0, o?.hi ?? c.data.t1];
export const curveOn = (o: CurveOpts | undefined): boolean => o?.on ?? true;

export class CurvesPane {
  private readonly body: HTMLElement;
  constructor(private readonly panel: HTMLElement) {
    this.body = panel.querySelector<HTMLElement>(".body")!;
  }

  /** build the rows; `opts` returns the host's current options, `onChange` receives the new options of one curve */
  build(curves: readonly Curve[], opts: () => Record<string, CurveOpts>, onChange: (id: string, o: CurveOpts) => void): void {
    this.body.replaceChildren();
    this.panel.style.display = curves.length ? "" : "none";
    for (const c of curves) {
      const [t0, t1] = c.data.interval;
      const cd = new Codomain(c.param.codomain ?? "lin");
      const fmt = (t: number) => `${formatReal(t, 3)}${cd.unit ? ` ${cd.unit}` : c.param.unit ? ` ${c.param.unit}` : ""}`;
      const row = document.createElement("div");
      row.className = "row curve";
      row.dataset.curve = c.id;
      const label = document.createElement("label");
      label.className = "toggle";
      label.textContent = c.name;
      const kind = c.data.kind === "sampled" ? `${c.data.sampleTimes!.length} samples` : c.spec.data.type === "flow" ? "an integral curve" : "an expression";
      label.dataset.tip = `${c.name}: ${kind}, ${c.param.name} ∈ [${fmt(t0)}, ${fmt(t1)}]. Click to show / hide the curve.`;
      label.addEventListener("click", () => onChange(c.id, { ...(opts()[c.id] ?? {}), on: !curveOn(opts()[c.id]) }));
      row.appendChild(label);
      const sl = document.createElement("div");
      sl.className = "isl";
      sl.dataset.min = String(t0); sl.dataset.max = String(t1);
      sl.dataset.step = String((t1 - t0) / 1000);
      const cur = opts()[c.id];
      sl.dataset.lo = cur?.lo == null ? "null" : String(cur.lo);
      sl.dataset.hi = cur?.hi == null ? "null" : String(cur.hi);
      sl.dataset.tip = `The range of ${c.param.name} that is drawn. Empty = the whole curve. Drag out a range, drag a handle or the band; click a handle to open that end; Backspace clears.`;
      row.appendChild(sl);
      const val = document.createElement("span");
      val.className = "val wide";
      row.appendChild(val);
      const info = document.createElement("span");
      info.className = "info";
      info.textContent = "i";
      bindInfoIcon(info, c.info);
      row.appendChild(info);
      const slider = makeIntervalSlider(sl) as IntervalEl;
      const paint = () => {
        const o = opts()[c.id];
        const [a, b] = [slider.lo ?? t0, slider.hi ?? t1];
        val.textContent = slider.lo === null && slider.hi === null ? "all" : `${fmt(a)} – ${fmt(b)}`;
        row.classList.toggle("off", !curveOn(o));
      };
      paint();
      slider.addEventListener("input", paint);
      slider.addEventListener("change", () => { paint(); onChange(c.id, { ...(opts()[c.id] ?? {}), lo: slider.lo, hi: slider.hi }); });
      // live preview while dragging: the host reads the slider through `preview`
      slider.addEventListener("input", () => this.onPreview?.(c.id, slider.lo, slider.hi));
      this.body.appendChild(row);
    }
    installTooltips(this.body);
  }

  /** the host's live-drag hook: the shown range follows the slider before it is committed */
  onPreview: ((id: string, lo: number | null, hi: number | null) => void) | undefined;

  /** repaint the on / off state (after an option change) */
  refresh(opts: Record<string, CurveOpts>): void {
    for (const row of this.body.querySelectorAll<HTMLElement>(".row.curve")) row.classList.toggle("off", !curveOn(opts[row.dataset.curve!]));
  }
}
