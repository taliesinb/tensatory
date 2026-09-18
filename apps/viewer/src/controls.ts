// The Controls pane: one row per control row of the bundle (core `controlRows`:
// random-direction rows of displaced nets and widgeted random arrays). A row
// holds ADJUSTMENTS, never edits the bundle: ↻ draws a fresh 32-bit salt that
// core hashes into every random array the row governs, the slider is a
// multiplier on the row's scale (log-spaced; unset = ×1). The host applies
// them by rebuilding the bundle from the adjusted spec (core `adjustSpec`):
// shapes never change, so the GPU programs keep their code (no recompiles)
// and only the packed constants differ. Rows are disabled while an animation
// plays — a rebuild mid-animation would stutter it.

import { freshSeed, type Adjustments, type ControlRow, type RowAdjustment } from "@tensatory/core";
import { installTooltips, makeSlider, type SliderEl } from "./widgets";

const fmtScale = (s: number): string => (s >= 100 ? `×${s.toFixed(0)}` : s >= 10 ? `×${s.toFixed(1)}` : s >= 1 ? `×${s.toFixed(2)}` : `×${s.toPrecision(2)}`);

export class ControlsPane {
  private readonly body: HTMLElement;
  private enabled = true;

  constructor(private readonly panel: HTMLElement) {
    this.body = panel.querySelector<HTMLElement>(".body")!;
  }

  /**
   * Build the rows. `adjust` returns the host's CURRENT adjustments (the host replaces the object on every change);
   * `onChange` receives the row id and its new adjustment (an empty object = back to the bundle's values) when the
   * user commits.
   */
  build(rows: readonly ControlRow[], adjust: () => Adjustments, onChange: (id: string, a: RowAdjustment) => void): void {
    this.body.replaceChildren();
    this.panel.style.display = rows.length ? "" : "none";
    for (const r of rows) {
      const row = document.createElement("div");
      row.className = "row ctl";
      const label = document.createElement("label");
      label.textContent = r.label;
      label.dataset.tip = r.kind === "direction"
        ? `A random direction of a displaced net (${r.members.length} random array${r.members.length === 1 ? "" : "s"}). ↻ draws new random arrays (same distributions, same length); the slider multiplies the direction. Both are remembered per bundle, the bundle itself is not changed.`
        : `A random array. ↻ redraws it (same distribution); the slider multiplies the distribution's scale. Remembered per bundle.`;
      row.appendChild(label);
      const current = adjust()[r.id] ?? {};
      if (r.hasScale) {
        const sl = document.createElement("div");
        sl.className = "sl";
        const [lo, hi] = r.scaleRange;
        sl.dataset.min = String(Math.log10(lo)); sl.dataset.max = String(Math.log10(hi));
        sl.dataset.step = String(r.scaleSteps ? (Math.log10(hi) - Math.log10(lo)) / r.scaleSteps : 0.01);
        sl.dataset.nullable = "";
        sl.dataset.value = current.scale === undefined || current.scale === 1 ? "null" : String(Math.log10(current.scale));
        sl.dataset.tip = `Scale multiplier, log-spaced over ${fmtScale(lo)} … ${fmtScale(hi)}. Click the tick (or Backspace) to unset: ×1, the bundle's own scale.`;
        row.appendChild(sl);
        const val = document.createElement("span");
        val.className = "val";
        row.appendChild(val);
        const slider = makeSlider(sl) as SliderEl;
        const scaleOf = () => (slider.value === null ? 1 : Math.pow(10, +slider.value));
        const paint = () => { val.textContent = fmtScale(scaleOf()); };
        paint();
        slider.addEventListener("input", paint);
        slider.addEventListener("change", () => {
          paint();
          const a = { ...(adjust()[r.id] ?? {}) };
          if (slider.value === null) delete a.scale; else a.scale = scaleOf();
          onChange(r.id, a);
        });
      } else {
        const spacer = document.createElement("span"); spacer.className = "hint"; spacer.style.flex = "1"; spacer.textContent = "—";
        row.appendChild(spacer);
        const val = document.createElement("span"); val.className = "val"; row.appendChild(val);
      }
      if (r.members.length) {
        const btn = document.createElement("button");
        btn.className = "tickbtn reseed";
        btn.textContent = "↻";
        btn.dataset.tip = "Reseed: draw new random arrays for this row (shift-click: back to the bundle's own seeds).";
        btn.addEventListener("click", (e) => {
          const a = { ...(adjust()[r.id] ?? {}) };
          if (e.shiftKey) delete a.seed; else a.seed = freshSeed();
          onChange(r.id, a);
        });
        row.appendChild(btn);
      }
      this.body.appendChild(row);
    }
    installTooltips(this.body);
    this.setEnabled(this.enabled);
  }

  /** rows are inert while an animation plays */
  setEnabled(on: boolean): void {
    this.enabled = on;
    this.panel.classList.toggle("inert", !on);
    this.panel.title = on ? "" : "controls are disabled while an animation plays (space pauses)";
  }
}
