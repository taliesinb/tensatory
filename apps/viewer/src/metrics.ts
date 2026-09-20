// The metrics table: one <svg>, rows = fields, columns = visualization slots.
// All logic from pointer coordinates (no per-element handlers).
// Shift-preview captures the column (or every set slot from the name column)
// and follows the pointer's y until shift is released; click locks; clicking
// the selected cell deselects it; wheel / arrows step the hovered column.
//
// Field names are PATHS: "train/loss/setosa" is a row under "train/loss"
// under "train". The tree is drawn flattened with a subtle indent; subtrees
// are collapsed until their disclosure marker is clicked, except that a
// LOCKED-selected field is always shown, with its ancestors, while its
// unselected siblings stay hidden. A path may be a field AND a parent
// ("train/loss" with "train/loss/setosa" below it); a path that is only a
// parent ("train") is a pure heading with no matrix cells.
//
// A field with a summary / details gets a ⓘ at the right edge of the name
// column: hover = the summary (else details) tooltip, click = the details modal
// (info.ts); the name is shortened to leave it room.

import type { Info } from "@tensatory/core";
import { hoverText, showInfo } from "./info";
import { installTooltips } from "./widgets";

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

export interface MetricsRow { id: string; name: string; kind: FieldKind; info?: Info | undefined }

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
  /** the table's height changed (a subtree opened / closed, a selection moved into a collapsed one) */
  onLayout?: () => void;
}

/** a node of the name tree: a field, a heading, or both */
interface Node {
  path: string;
  label: string;
  depth: number;
  id?: string;
  kind?: FieldKind;
  info?: Info | undefined;
  children: Node[];
}

/** a drawn row */
interface Line { node: Node; forced: boolean }

const SEP = "/";

export class MetricsTable {
  private roots: Node[] = [];
  private lines: Line[] = [];
  private cols: SlotKey[] = [];
  private prev: { col: SlotKey | "all" } | null = null;
  private over = false;
  private last: PointerEvent | null = null;
  /** expanded headings (by path); survives rebuilds so a space switch keeps the tree as the user left it */
  private readonly expanded = new Set<string>();
  private readonly geo = { rowH: 15, headH: 20, nameW: 172, cellW: 24, pad: 4, paneW: 282, indent: 9, infoW: 16 };

  constructor(private readonly o: MetricsTableOptions) {
    if (o.paneW) this.geo.paneW = o.paneW;
    window.addEventListener("keydown", (e) => { if (e.key === "Shift" && this.over && this.last && this.lines.length) this.start(this.last); });
    window.addEventListener("keyup", (e) => { if (e.key === "Shift") this.end(); });
    window.addEventListener("blur", () => this.end());
    window.addEventListener("keydown", (e) => {
      if (!this.over || !this.last || e.shiftKey || !this.lines.length) return;
      if (e.key === "ArrowDown") { e.preventDefault(); this.step(this.last, 1); } else if (e.key === "ArrowUp") { e.preventDefault(); this.step(this.last, -1); }
    });
  }

  /** slots whose panel is currently enabled (disabled columns stay visible, dimmed) */
  get visibleSlots(): SlotKey[] { return this.o.slots.filter((s) => (s.present?.() ?? true) && s.visible()).map((s) => s.key); }
  private get allSlotKeys(): SlotKey[] { return this.o.slots.filter((s) => s.present?.() ?? true).map((s) => s.key); }

  /*******************************************************/
  /* tree */

  private static tree(rows: MetricsRow[]): Node[] {
    const roots: Node[] = [];
    const byPath = new Map<string, Node>();
    for (const r of rows) {
      const parts = r.name.split(SEP).map((p) => p.trim()).filter((p) => p.length);
      if (!parts.length) parts.push(r.name);
      let siblings = roots, path = "";
      parts.forEach((label, depth) => {
        path = path ? `${path}${SEP}${label}` : label;
        let node = byPath.get(path);
        if (!node) { node = { path, label, depth, children: [] }; byPath.set(path, node); siblings.push(node); }
        if (depth === parts.length - 1) {
          if (node.id !== undefined) { // two fields with one name: keep both as siblings (the second under its id)
            node = { path: `${path}${SEP}#${r.id}`, label, depth, children: [] }; siblings.push(node);
          }
          node.id = r.id; node.kind = r.kind; node.info = r.info;
        }
        siblings = node.children;
      });
    }
    return roots;
  }

  /** the rows to draw: expanded subtrees whole, collapsed ones only their locked-selected descendants (with ancestors) */
  private layout(): Line[] {
    const locked = new Set(Object.values(this.o.lockedSel()).filter((v): v is string => v !== null));
    const hasLocked = (n: Node): boolean => (n.id !== undefined && locked.has(n.id)) || n.children.some(hasLocked);
    const out: Line[] = [];
    // under a collapsed ancestor only the path to a locked field is shown (its own expansion does not reopen it)
    const visit = (n: Node, forced: boolean) => {
      out.push({ node: n, forced });
      if (!n.children.length) return;
      if (!forced && this.expanded.has(n.path)) n.children.forEach((c) => visit(c, false));
      else for (const c of n.children) if (hasLocked(c)) visit(c, true);
    };
    this.roots.forEach((n) => visit(n, false));
    return out;
  }

  /** the field rows, in drawn order */
  private get fieldLines(): Line[] { return this.lines.filter((l) => l.node.id !== undefined); }
  private nameX(n: Node): number { return this.geo.pad + this.geo.indent + n.depth * this.geo.indent; }

  /*******************************************************/
  /* pointer logic */

  private hit(e: PointerEvent | WheelEvent): { line: Line; col: SlotKey | "all"; header: boolean; marker: boolean; info: boolean } | null {
    const svg = this.o.body.querySelector("svg");
    if (!svg || !this.lines.length) return null;
    const r = svg.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top, g = this.geo;
    const header = y < g.headH;
    const row = Math.max(0, Math.min(this.lines.length - 1, Math.floor((y - g.headH) / g.rowH)));
    const line = this.lines[row]!;
    const col = x < g.nameW ? "all" : this.cols[Math.max(0, Math.min(this.cols.length - 1, Math.floor((x - g.nameW) / g.cellW)))]!;
    const info = !header && line.node.info !== undefined && x >= g.nameW - g.infoW && x < g.nameW;
    return { line, col, header, marker: x < this.nameX(line.node), info };
  }

  /** name-column targets: the visible slots that are set, or all of them */
  private allSlots(id: string, everything = false): Sel {
    const locked = this.o.lockedSel();
    return Object.fromEntries(this.visibleSlots.filter((k) => everything || locked[k] !== NONE).map((k) => [k, id]));
  }
  private apply(h: { line: Line; col: SlotKey | "all" }): void {
    const id = h.line.node.id;
    if (id === undefined) return; // a heading: the preview keeps its previous row
    this.o.setSel(this.prev!.col === "all" ? this.allSlots(id) : { [this.prev!.col]: id }, false);
  }
  private readonly global = (e: PointerEvent) => { if (!e.shiftKey) { this.end(); return; } const h = this.hit(e); if (h) this.apply(h); };
  private start(e: PointerEvent): void { const h = this.hit(e); if (this.prev || !h || h.header || h.info) return; this.prev = { col: h.col }; window.addEventListener("pointermove", this.global); this.apply(h); }
  private end(): void { if (!this.prev) return; this.prev = null; window.removeEventListener("pointermove", this.global); this.o.setSel({ ...this.o.lockedSel() }, false); }

  private step(e: PointerEvent | WheelEvent, dir: number): void {
    const h = this.hit(e); if (!h) return;
    const cols = h.col === "all" ? this.visibleSlots : [h.col];
    const fields = this.fieldLines;
    if (!fields.length) return;
    const locked = this.o.lockedSel(), sel: Sel = {};
    for (const k of cols) {
      const r = fields.findIndex((l) => l.node.id === locked[k]);
      const nr = Math.max(0, Math.min(fields.length - 1, (r < 0 ? 0 : r) + dir));
      sel[k] = fields[nr]!.node.id!;
    }
    this.o.setSel(sel, true);
  }

  private toggle(n: Node): void {
    if (this.expanded.has(n.path)) this.expanded.delete(n.path); else this.expanded.add(n.path);
    this.draw();
  }

  /*******************************************************/
  /* drawing */

  build(rows: MetricsRow[]): void {
    this.roots = MetricsTable.tree(rows);
    this.cols = this.allSlotKeys;
    this.geo.nameW = this.geo.paneW - this.cols.length * this.geo.cellW;
    this.draw();
  }

  private draw(): void {
    const body = this.o.body; body.replaceChildren();
    const cols = this.cols, g = this.geo;
    const lines = (this.lines = this.layout());
    const W = g.paneW, H = g.headH + lines.length * g.rowH;
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
    lines.forEach(({ node, forced }, r) => {
      const y = g.headH + r * g.rowH, x = this.nameX(node);
      const heading = node.id === undefined;
      if (node.children.length) {
        const open = !forced && this.expanded.has(node.path);
        // "+" = click to expand, "−" (U+2212, full width) = click to collapse; a collapsed subtree that still shows
        // its selected descendants beneath is dimmer
        const dim = !open && lines[r + 1]?.forced === true && lines[r + 1]!.node.depth > node.depth;
        el("text", { x: x - g.indent + 4, y: y + g.rowH - 4, fill: dim ? "#6a7690" : "#8a96b0", "font-size": 11, "text-anchor": "middle", "pointer-events": "none" }, open ? "\u2212" : "+");
      }
      const maxChars = Math.max(6, Math.floor((g.nameW - x - 2 - (node.info ? g.infoW : 0)) / 6.4));
      const name = node.label.length > maxChars ? `${node.label.slice(0, maxChars - 1)}…` : node.label;
      const t = el("text", { x, y: y + g.rowH - 4, fill: heading ? "#7c869c" : "#aab" }, name);
      if (node.label !== node.path) { const title = document.createElementNS(ns, "title"); title.textContent = node.path; t.appendChild(title); }
      if (node.info) {
        // ⓘ at the right edge of the name column; its tooltip goes through the shared [data-tip] mechanism
        const cx = g.nameW - g.infoW / 2 - 1, cy = y + g.rowH / 2;
        const grp = document.createElementNS(ns, "g");
        grp.dataset.tip = hoverText(node.info); grp.style.cursor = "help"; grp.setAttribute("class", "info");
        const c = document.createElementNS(ns, "circle"); c.setAttribute("cx", String(cx)); c.setAttribute("cy", String(cy)); c.setAttribute("r", "5"); c.setAttribute("fill", "transparent"); c.setAttribute("stroke", "#6a7690"); c.setAttribute("stroke-width", "1");
        const i = document.createElementNS(ns, "text"); i.setAttribute("x", String(cx)); i.setAttribute("y", String(cy + 3)); i.setAttribute("text-anchor", "middle"); i.setAttribute("font-size", "8"); i.setAttribute("font-style", "italic"); i.setAttribute("font-family", "Georgia, 'Times New Roman', serif"); i.setAttribute("fill", "#8a96b0"); i.setAttribute("pointer-events", "none"); i.textContent = "i";
        grp.append(c, i); svg.appendChild(grp);
      }
      if (heading) return;
      cols.forEach((k, c) => {
        el("rect", { x: g.nameW + c * g.cellW + 1.5, y: y + 1.5, width: g.cellW - 3, height: g.rowH - 3, fill: "#232a3a", "data-slot": k, "data-id": node.id! });
        const def = this.o.slots.find((s) => s.key === k)!;
        const glyph = useGlyph(def.type, node.kind!);
        if (glyph) el("text", { x: g.nameW + c * g.cellW + g.cellW / 2, y: y + g.rowH - 4, fill: "#5c6a88", "text-anchor": "middle", "font-size": 9.5, "pointer-events": "none", "data-glyph": k, "data-id": node.id! }, glyph);
      });
    });
    svg.style.cursor = "pointer";
    svg.addEventListener("pointerenter", (e) => { this.over = true; this.last = e; });
    svg.addEventListener("pointerleave", () => { this.over = false; });
    svg.addEventListener("pointermove", (e) => { this.last = e; });
    svg.addEventListener("pointerdown", (e) => {
      const h = this.hit(e); this.end(); if (!h) return;
      if (h.info) { showInfo(h.line.node.info!); return; }
      if (h.header) { if (h.col !== "all") this.o.slots.find((s) => s.key === h.col)?.toggle?.(); return; }
      const n = h.line.node;
      // headings toggle from anywhere in the name column; a field that is also a parent toggles from its marker only
      if (n.children.length && h.col === "all" && (n.id === undefined || h.marker)) { this.toggle(n); return; }
      if (n.id === undefined) return;
      const locked = this.o.lockedSel();
      if (h.col === "all") {
        const set = this.visibleSlots.filter((k) => locked[k] !== NONE);
        const already = set.length > 0 && set.every((k) => locked[k] === n.id);
        this.o.setSel(this.allSlots(n.id, already || set.length === 0), true);
      } else this.o.setSel({ [h.col]: locked[h.col] === n.id ? NONE : n.id }, true);
    });
    svg.addEventListener("wheel", (e) => { if (e.shiftKey) return; e.preventDefault(); if (e.deltaY && !this.hit(e)?.header) this.step(e, e.deltaY > 0 ? 1 : -1); }, { passive: false });
    body.appendChild(svg);
    installTooltips(svg);
    this.paint();
    this.o.onLayout?.();
  }

  /** the drawn rows changed when a locked selection moved into or out of a collapsed subtree */
  refresh(): void {
    const want = this.layout();
    if (want.length !== this.lines.length || want.some((l, i) => l.node !== this.lines[i]!.node || l.forced !== this.lines[i]!.forced)) { this.draw(); return; }
    this.paint();
  }

  private paint(): void {
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
