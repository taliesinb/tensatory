// The metrics table: one <svg>, rows = scalar fields, columns = visualization
// slots. All logic from pointer coordinates (no per-element handlers).
// Shift-preview captures the column (or every set slot from the name column)
// and follows the pointer's y until shift is released; click locks; clicking
// the selected cell deselects it; wheel / arrows step the hovered column.

export type SlotKey = string;
export const NONE = null;
export type Sel = Record<SlotKey, string | null>;

export type FieldKind = "scalar" | "vector";

export interface SlotDef {
  key: SlotKey;
  /** header text; `sub` is rendered as a subscript */
  label: string;
  sub?: string;
  tip: string;
  /** what the slot consumes: a scalar slot set to a vector field uses its norm, a vector slot set to a scalar field uses its gradient */
  type: FieldKind;
  visible: () => boolean;
  /** whether the column exists at all in the current space (default: always); absent columns are not drawn */
  present?: () => boolean;
  /** clicking the column header enables / disables the slot's group (like the panel's tick) */
  toggle?: () => void;
}

export interface MetricsRow { id: string; name: string; kind: FieldKind }

/** how a slot of `type` would use a field of `kind` */
export const useGlyph = (type: FieldKind, kind: FieldKind): string => (type === kind ? "" : type === "vector" ? "∇" : "|·|");

export interface MetricsTableOptions {
  body: HTMLElement;
  slots: SlotDef[];
  /** called with the slots to set; lock=false is a hover preview */
  setSel: (partial: Sel, lock: boolean) => void;
  sel: () => Sel;
  lockedSel: () => Sel;
  paneW?: number;
}

export class MetricsTable {
  private rows: MetricsRow[] = [];
  private cols: SlotKey[] = [];
  private prev: { col: SlotKey | "all" } | null = null;
  private over = false;
  private last: PointerEvent | null = null;
  private readonly geo = { rowH: 15, headH: 20, nameW: 172, cellW: 24, pad: 4, paneW: 282 };

  constructor(private readonly o: MetricsTableOptions) {
    if (o.paneW) this.geo.paneW = o.paneW;
    window.addEventListener("keydown", (e) => { if (e.key === "Shift" && this.over && this.last && this.rows.length) this.start(this.last); });
    window.addEventListener("keyup", (e) => { if (e.key === "Shift") this.end(); });
    window.addEventListener("blur", () => this.end());
    window.addEventListener("keydown", (e) => {
      if (!this.over || !this.last || e.shiftKey || !this.rows.length) return;
      if (e.key === "ArrowDown") { e.preventDefault(); this.step(this.last, 1); } else if (e.key === "ArrowUp") { e.preventDefault(); this.step(this.last, -1); }
    });
  }

  /** slots whose panel is currently enabled (disabled columns stay visible, dimmed) */
  get visibleSlots(): SlotKey[] { return this.o.slots.filter((s) => (s.present?.() ?? true) && s.visible()).map((s) => s.key); }
  private get allSlotKeys(): SlotKey[] { return this.o.slots.filter((s) => s.present?.() ?? true).map((s) => s.key); }

  private hit(e: PointerEvent | WheelEvent): { id: string; col: SlotKey | "all"; header: boolean } | null {
    const svg = this.o.body.querySelector("svg");
    if (!svg || !this.rows.length) return null;
    const r = svg.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top, g = this.geo;
    const header = y < g.headH;
    const row = Math.max(0, Math.min(this.rows.length - 1, Math.floor((y - g.headH) / g.rowH)));
    const col = x < g.nameW ? "all" : this.cols[Math.max(0, Math.min(this.cols.length - 1, Math.floor((x - g.nameW) / g.cellW)))]!;
    return { id: this.rows[row]!.id, col, header };
  }

  /** name-column targets: the visible slots that are set, or all of them */
  private allSlots(id: string, everything = false): Sel {
    const locked = this.o.lockedSel();
    return Object.fromEntries(this.visibleSlots.filter((k) => everything || locked[k] !== NONE).map((k) => [k, id]));
  }
  private apply(h: { id: string; col: SlotKey | "all" }): void {
    this.o.setSel(this.prev!.col === "all" ? this.allSlots(h.id) : { [this.prev!.col]: h.id }, false);
  }
  private readonly global = (e: PointerEvent) => { if (!e.shiftKey) { this.end(); return; } const h = this.hit(e); if (h) this.apply(h); };
  private start(e: PointerEvent): void { const h = this.hit(e); if (this.prev || !h || h.header) return; this.prev = { col: h.col }; window.addEventListener("pointermove", this.global); this.apply(h); }
  private end(): void { if (!this.prev) return; this.prev = null; window.removeEventListener("pointermove", this.global); this.o.setSel({ ...this.o.lockedSel() }, false); }

  private step(e: PointerEvent | WheelEvent, dir: number): void {
    const h = this.hit(e); if (!h) return;
    const cols = h.col === "all" ? this.visibleSlots : [h.col];
    const locked = this.o.lockedSel(), sel: Sel = {};
    for (const k of cols) {
      const r = this.rows.findIndex((row) => row.id === locked[k]);
      const nr = Math.max(0, Math.min(this.rows.length - 1, (r < 0 ? 0 : r) + dir));
      sel[k] = this.rows[nr]!.id;
    }
    this.o.setSel(sel, true);
  }

  build(rows: MetricsRow[]): void {
    this.rows = rows;
    const body = this.o.body; body.replaceChildren();
    const cols = this.allSlotKeys; this.cols = cols;
    const g = this.geo;
    g.nameW = g.paneW - cols.length * g.cellW;
    const W = g.paneW, H = g.headH + rows.length * g.rowH;
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("width", String(W)); svg.setAttribute("height", String(H)); svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const el = (tag: string, attrs: Record<string, string | number>, text?: string) => {
      const e = document.createElementNS(ns, tag);
      for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
      if (text != null) e.textContent = text;
      svg.appendChild(e); return e;
    };
    cols.forEach((k, c) => {
      const def = this.o.slots.find((s) => s.key === k)!;
      const t = el("text", { x: g.nameW + c * g.cellW + g.cellW / 2, y: g.headH - 9, fill: "#9ab", "text-anchor": "middle", "font-size": 11, "data-head": k });
      t.textContent = def.label;
      if (def.sub) { const sub = document.createElementNS(ns, "tspan"); sub.setAttribute("baseline-shift", "sub"); sub.setAttribute("font-size", "8"); sub.textContent = def.sub; t.appendChild(sub); }
      const title = document.createElementNS(ns, "title"); title.textContent = def.tip; t.appendChild(title);
    });
    rows.forEach((row, r) => {
      const y = g.headH + r * g.rowH;
      const name = row.name.length > 26 ? `${row.name.slice(0, 25)}…` : row.name;
      el("text", { x: g.pad, y: y + g.rowH - 4, fill: "#aab" }, name);
      cols.forEach((k, c) => {
        el("rect", { x: g.nameW + c * g.cellW + 1.5, y: y + 1.5, width: g.cellW - 3, height: g.rowH - 3, fill: "#232a3a", "data-slot": k, "data-id": row.id });
        const def = this.o.slots.find((s) => s.key === k)!;
        const glyph = useGlyph(def.type, row.kind);
        if (glyph) el("text", { x: g.nameW + c * g.cellW + g.cellW / 2, y: y + g.rowH - 4, fill: "#5c6a88", "text-anchor": "middle", "font-size": 9.5, "pointer-events": "none", "data-glyph": k, "data-id": row.id }, glyph);
      });
    });
    svg.style.cursor = "pointer";
    svg.addEventListener("pointerenter", (e) => { this.over = true; this.last = e; });
    svg.addEventListener("pointerleave", () => { this.over = false; });
    svg.addEventListener("pointermove", (e) => { this.last = e; });
    svg.addEventListener("pointerdown", (e) => {
      const h = this.hit(e); this.end(); if (!h) return;
      if (h.header) { if (h.col !== "all") this.o.slots.find((s) => s.key === h.col)?.toggle?.(); return; }
      const locked = this.o.lockedSel();
      if (h.col === "all") {
        const set = this.visibleSlots.filter((k) => locked[k] !== NONE);
        const already = set.length > 0 && set.every((k) => locked[k] === h.id);
        this.o.setSel(this.allSlots(h.id, already || set.length === 0), true);
      } else this.o.setSel({ [h.col]: locked[h.col] === h.id ? NONE : h.id }, true);
    });
    svg.addEventListener("wheel", (e) => { if (e.shiftKey) return; e.preventDefault(); if (e.deltaY && !this.hit(e)?.header) this.step(e, e.deltaY > 0 ? 1 : -1); }, { passive: false });
    body.appendChild(svg);
    this.refresh();
  }

  refresh(): void {
    const sel = this.o.sel(), locked = this.o.lockedSel();
    const enabled = new Set(this.visibleSlots);
    for (const h of this.o.body.querySelectorAll<SVGTextElement>("text[data-head]")) h.setAttribute("fill", enabled.has(h.getAttribute("data-head")!) ? "#9ab" : "#4a5266");
    for (const r of this.o.body.querySelectorAll<SVGRectElement>("rect[data-slot]")) {
      const k = r.getAttribute("data-slot")!, id = r.getAttribute("data-id")!;
      const on = enabled.has(k);
      r.setAttribute("fill", sel[k] === id ? (on ? "#3b82f6" : "#2a3f66") : locked[k] === id ? (on ? "#1e3a6e" : "#1b2740") : on ? "#232a3a" : "#1a1f2b");
    }
    for (const t of this.o.body.querySelectorAll<SVGTextElement>("text[data-glyph]")) {
      const k = t.getAttribute("data-glyph")!, id = t.getAttribute("data-id")!;
      const on = enabled.has(k);
      t.setAttribute("fill", sel[k] === id ? (on ? "#e8f0ff" : "#7f8db0") : locked[k] === id ? (on ? "#9fb3d9" : "#5a6a8c") : on ? "#5c6a88" : "#3a4358");
    }
  }
}
