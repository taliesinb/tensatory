// The TABLE PICKER of the bundle panel's `bundle` and `space` rows: a themed replacement for the OS-native <select>.
// Closed, it is a frameless control (the current name as plain text, flush with the row's key, a ⌃⌄ chevron at the
// right) that steps with the wheel / ↑↓ like the
// select did (main.ts `stepOnWheel`); clicked, it opens a popover TABLE under itself — one row per alternative with
// its name and a few descriptive columns (spaces, fields, a truncated summary; `describe*` below format them from a
// core `Inventory`), the current row ticked, hover / ↑↓ highlight, Enter or a click picks, Escape / a click outside /
// a scroll closes. A picker with a single alternative is a plain label (nothing to pick from), its ⓘ stays.
// Rows are produced on demand (`rowsOf`), so a table opened before the documents' inventories have arrived shows "…"
// and `refresh()` redraws it when they do.

import type { Inventory } from "@tensatory/core";
import { installTooltips } from "./widgets";

export interface PickerColumn { title: string; cls?: string }
export interface PickerRow {
  id: string;
  /** what the closed control shows for this row; defaults to the first cell's text */
  label?: string;
  /** one entry per column: text, or a built node (a name with a tag beside it) */
  cells: (string | Node)[];
  /** per column: a tooltip (the full text of a truncated cell) */
  tips?: (string | undefined)[];
}

const CHEVRON = `<svg viewBox="0 0 10 16" width="8" height="13" aria-hidden="true"><path d="M2 6l3-3.5L8 6M2 10l3 3.5L8 10" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>`;

export class TablePicker {
  /** the closed control, in the row */
  readonly el: HTMLElement;
  private readonly label: HTMLElement;
  private readonly pop: HTMLElement;
  private rows: PickerRow[] = [];
  private value = "";
  private hi = -1;
  onPick: ((id: string) => void) | undefined;

  /** `host` is the element the picker replaces (the former <select>); its id moves to the control */
  constructor(host: HTMLElement, private columns: PickerColumn[], private readonly rowsOf: () => PickerRow[]) {
    this.el = document.createElement("div");
    this.el.className = "tpick-btn";
    this.el.id = host.id;
    this.label = document.createElement("span"); this.label.className = "lbl";
    const chev = document.createElement("span"); chev.className = "chev"; chev.innerHTML = CHEVRON;
    this.el.append(this.label, chev);
    host.replaceWith(this.el);
    this.pop = document.createElement("div"); this.pop.className = "tpick"; this.pop.style.display = "none";
    document.body.appendChild(this.pop);
    this.el.addEventListener("click", () => { if (this.el.classList.contains("single")) return; this.isOpen ? this.close() : this.open(); });
    // outside: any pointerdown that is not on the control or in the table closes it; so does a scroll anywhere (the
    // popover is anchored to the control's screen position) and the window losing focus or changing size
    window.addEventListener("pointerdown", (e) => { if (this.isOpen && !this.pop.contains(e.target as Node) && !this.el.contains(e.target as Node)) this.close(); }, true);
    window.addEventListener("scroll", (e) => { if (this.isOpen && !this.pop.contains(e.target as Node)) this.close(); }, true);
    window.addEventListener("resize", () => this.close());
    window.addEventListener("blur", () => this.close());
    window.addEventListener("keydown", (e) => {
      if (!this.isOpen) return;
      if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); this.close(); return; }
      if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); e.stopPropagation(); this.highlight(this.hi + (e.key === "ArrowDown" ? 1 : -1)); return; }
      if (e.key === "Home" || e.key === "End") { e.preventDefault(); e.stopPropagation(); this.highlight(e.key === "Home" ? 0 : this.rows.length - 1); return; }
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); e.stopPropagation(); const r = this.rows[this.hi]; this.close(); if (r) this.pick(r.id); return; }
      // any other key is the page's (r = refit, c / i / s / v = panels): the table simply closes first
      this.close();
    }, true);
  }

  get isOpen(): boolean { return this.pop.style.display !== "none"; }

  /** replace the columns (a picker whose alternatives' descriptive columns depend on the data: a sweep's keys) */
  setColumns(columns: PickerColumn[]): void { this.columns = columns; if (this.isOpen) this.build(); }

  /** the current alternative; `fallback` is shown when `id` is not among the rows (a bundle opened from disk) */
  set(id: string, fallback = id): void {
    this.value = id;
    this.rows = this.rowsOf();
    const row = this.rows.find((r) => r.id === id);
    this.label.textContent = row ? row.label ?? textOf(row.cells[0]!) : fallback;
    this.el.classList.toggle("single", this.rows.length < 2);
    if (this.isOpen) this.build();
  }
  get current(): string { return this.value; }
  /** how many alternatives there are */
  get count(): number { return this.rows.length; }

  /** the alternatives changed (an inventory arrived): redraw the control and, when open, the table */
  refresh(): void { this.set(this.value, this.label.textContent ?? this.value); }

  /** step to the neighbouring alternative (wheel / ↑↓ over the closed control), wrapping */
  step(dir: number): void {
    this.rows = this.rowsOf();
    const n = this.rows.length; if (n < 2) return;
    const i = this.rows.findIndex((r) => r.id === this.value);
    this.pick(this.rows[(((i < 0 ? 0 : i + dir) % n) + n) % n]!.id);
  }

  private pick(id: string): void { this.onPick?.(id); }

  open(): void {
    this.rows = this.rowsOf();
    if (this.rows.length < 2) return;
    this.pop.style.display = "";
    this.el.classList.add("open");
    this.build();
  }
  close(): void {
    if (!this.isOpen) return;
    this.pop.style.display = "none";
    this.el.classList.remove("open");
    this.hi = -1;
  }

  private highlight(i: number): void {
    const n = this.rows.length; if (!n) return;
    this.hi = ((i % n) + n) % n;
    const trs = this.pop.querySelectorAll("tr.pick");
    trs.forEach((tr, k) => tr.classList.toggle("hi", k === this.hi));
    (trs[this.hi] as HTMLElement | undefined)?.scrollIntoView({ block: "nearest" });
  }

  /** (re)build the table and place it under the control (above when it does not fit below), within the viewport */
  private build(): void {
    const table = document.createElement("table");
    const head = table.createTHead().insertRow();
    head.insertCell(); // the ✓ column
    for (const c of this.columns) { const th = document.createElement("th"); th.textContent = c.title; if (c.cls) th.className = c.cls; head.appendChild(th); }
    const body = table.createTBody();
    this.rows.forEach((r, k) => {
      const tr = body.insertRow(); tr.className = "pick"; tr.dataset.id = r.id;
      if (r.id === this.value) tr.classList.add("on");
      if (k === this.hi) tr.classList.add("hi");
      const mark = tr.insertCell(); mark.className = "mark"; mark.textContent = r.id === this.value ? "✓" : "";
      r.cells.forEach((cell, i) => {
        const td = tr.insertCell(); td.className = this.columns[i]?.cls ?? "";
        if (typeof cell === "string") td.textContent = cell; else td.appendChild(cell);
        const tip = r.tips?.[i]; if (tip) td.dataset.tip = tip;
      });
      tr.addEventListener("pointerenter", () => this.highlight(k));
      tr.addEventListener("click", () => { this.close(); this.pick(r.id); });
    });
    this.pop.replaceChildren(table);
    installTooltips(this.pop);
    // position: left edge on the control's, below it; clamped to the viewport with a 8 px margin, above when below does not fit
    const r = this.el.getBoundingClientRect(), M = 8;
    this.pop.style.left = "0px"; this.pop.style.top = "0px"; this.pop.style.maxHeight = `${window.innerHeight - 2 * M}px`;
    const w = this.pop.offsetWidth, h = this.pop.offsetHeight;
    let x = r.left, y = r.bottom + 4;
    if (x + w > window.innerWidth - M) x = Math.max(M, window.innerWidth - M - w);
    if (y + h > window.innerHeight - M) { const above = r.top - 4 - h; y = above >= M ? above : Math.max(M, window.innerHeight - M - h); }
    this.pop.style.left = `${x}px`; this.pop.style.top = `${y}px`;
  }
}

const textOf = (c: string | Node): string => (typeof c === "string" ? c : c.textContent ?? "");

/*******************************************************/
/* the descriptive columns */

const SUP = "⁰¹²³⁴⁵⁶⁷⁸⁹";
const sup = (n: number): string => String(n).split("").map((d) => SUP[+d] ?? d).join("");
/** ℝ³ (ℝ⁰ for a dimension nothing declares) */
export const realsOf = (dims: number): string => `ℝ${sup(dims)}`;

/** "3 × ℝ³, 2 × ℝ⁵" (a lone space of a dimension without its count); "" for none */
export function describeSpaces(inv: Inventory): string {
  return inv.spaces.map(({ dims, count }) => `${count === 1 ? "" : `${count} × `}${realsOf(dims)}`).join(", ");
}

/** "5 scalar, 2 vector, 1 curve" (zero counts omitted; "no fields" for none) */
export function describeFields(inv: Inventory): string {
  const parts: string[] = [];
  if (inv.scalars) parts.push(`${inv.scalars} scalar`);
  if (inv.vectors) parts.push(`${inv.vectors} vector`);
  if (inv.curves) parts.push(`${inv.curves} curve${inv.curves === 1 ? "" : "s"}`);
  return parts.length ? parts.join(", ") : "no fields";
}

/**
 * Several inventories (a sweep's members) described together: where the members agree the plain count, where they
 * differ a range ("1–2 scalar", "ℝ², 0–1 × ℝ³" — a dimension some members lack ranges from 0).
 */
export function describeSpacesOf(invs: Inventory[]): string {
  const dims = [...new Set(invs.flatMap((i) => i.spaces.map((s) => s.dims)))].sort((a, b) => a - b);
  return dims.map((d) => {
    const counts = invs.map((i) => i.spaces.find((s) => s.dims === d)?.count ?? 0);
    const lo = Math.min(...counts), hi = Math.max(...counts);
    return `${lo === 1 && hi === 1 ? "" : `${range(lo, hi)} × `}${realsOf(d)}`;
  }).join(", ");
}
export function describeFieldsOf(invs: Inventory[]): string {
  const r = (pick: (i: Inventory) => number) => { const v = invs.map(pick); return [Math.min(...v), Math.max(...v)] as const; };
  const parts: string[] = [];
  const [s0, s1] = r((i) => i.scalars), [v0, v1] = r((i) => i.vectors), [c0, c1] = r((i) => i.curves);
  if (s1) parts.push(`${range(s0, s1)} scalar`);
  if (v1) parts.push(`${range(v0, v1)} vector`);
  if (c1) parts.push(`${range(c0, c1)} curve${c1 === 1 ? "" : "s"}`);
  return parts.length ? parts.join(", ") : "no fields";
}
const range = (lo: number, hi: number): string => (lo === hi ? `${lo}` : `${lo}–${hi}`);

/** a one-line summary cut to `max` characters (an ellipsis replacing the rest, at a word boundary when one is near) */
export function truncate(s: string, max = 60): string {
  const t = s.replace(/\s+/g, " ").trim();
  if (t.length <= max) return t;
  const cut = t.slice(0, max - 1);
  const sp = cut.lastIndexOf(" ");
  return `${(sp > max * 0.6 ? cut.slice(0, sp) : cut).replace(/[\s,;:—–-]+$/, "")}…`;
}
