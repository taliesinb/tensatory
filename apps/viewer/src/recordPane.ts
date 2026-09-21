// The MEMBER rows of the bundle panel (notes/sweeps.md §2): when the loaded document is a sweep, a `member` row
// between the `bundle` and `space` pickers shows which member is on view and moves through the sweep:
//
//   member    mlp · pca                         ⌃⌄  ⓘ    a table picker over the members (their keys as columns) + ⓘ
//             model: [mlp]  convnet                       one line per VARYING key: an inline label and a flipper —
//             dirs:  [pca]  random                        blue = this member's value, grey = a DIRECT switch (only
//                                                         this key changes), dim = other keys change too
//
// The ⓘ opens the member's summary / details with its whole RECORD as a table below (keys the flippers show, the
// ones that do not vary, and the member's attributes). Nothing here knows what a key means: core's `facets`
// discovers the values and their reachability from the records, `nearestMember` picks the member a click leads to.
// Keys the current member's record lacks are simply not shown (a convnet has no `num_layers`).

import { Codomain, formatReal, type Facet, type Info, type Sweep } from "@tensatory/core";
import type { RecordValue } from "@tensatory/schema";
import { bindInfoIcon, optionText, type InfoTable } from "./info";
import { TablePicker, truncate, type PickerColumn, type PickerRow } from "./picker";
import { installTooltips, stepOnWheel } from "./widgets";

/** a record value for display: ordinal keys through their codomain, integers in full (thin-space groups), other numbers compactly, booleans as words */
export function formatValue(v: RecordValue, f?: Facet): string {
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "number") {
    if (f?.spec?.codomain) return new Codomain(f.spec.codomain).format(v);
    if (Number.isInteger(v) && Math.abs(v) < 1e9) return String(Math.abs(v)).replace(/\B(?=(\d{3})+(?!\d))/g, "\u202f").replace(/^/, v < 0 ? "−" : "");
    return formatReal(v, 4);
  }
  return v;
}

export class RecordPane {
  private readonly picker: TablePicker;
  private readonly memberRow: HTMLElement;
  private readonly info: HTMLElement;
  private readonly lines: HTMLElement;
  private sweep: Sweep | undefined;
  private facets: Facet[] = [];
  onPick: ((id: string, why: string) => void) | undefined;
  /** what is known about a member NOT loaded yet (its document's name / summary, from the inventory prefetch) */
  describe: ((id: string) => { name?: string | undefined; summary?: string | undefined } | undefined) | undefined;

  constructor(private readonly host: HTMLElement) {
    this.memberRow = document.createElement("div"); this.memberRow.className = "prow";
    const label = document.createElement("label"); label.textContent = "member"; label.dataset.tip = "";
    const pickHost = document.createElement("div"); pickHost.id = "pickMember";
    this.info = document.createElement("span"); this.info.className = "info"; this.info.textContent = "i"; this.info.dataset.tip = "";
    this.memberRow.append(label, pickHost, this.info);
    this.lines = document.createElement("div");
    host.append(this.memberRow, this.lines);
    // the member table: one row per member — its name (the loaded bundle's, else the member id), the varying keys, the summary
    this.picker = new TablePicker(pickHost, [{ title: "member", cls: "name" }], () => this.rows());
    this.picker.onPick = (id) => this.onPick?.(id, `member → ${this.nameOf(id)}`);
    stepOnWheel(this.picker.el, (dir) => this.picker.step(dir));
    installTooltips(this.memberRow);
  }

  /** a member's display name: its bundle's name once loaded (or once its document was prefetched), else its id */
  private nameOf(id: string): string { return this.sweep?.loaded(id)?.name ?? this.describe?.(id)?.name ?? id; }
  private summaryOf(id: string): string { const b = this.sweep?.loaded(id); return b ? optionText(b.info) : this.describe?.(id)?.summary ?? ""; }

  /** what is known about the members changed (a prefetch landed): redraw the picker */
  refresh(): void { this.picker.refresh(); }

  private varying(): Facet[] { return this.facets.filter((f) => f.varying && !f.attribute); }

  private rows(): PickerRow[] {
    const sweep = this.sweep; if (!sweep) return [];
    const keys = this.varying();
    return sweep.memberIds.map((id) => {
      const rec = sweep.record(id);
      const summary = this.summaryOf(id), cut = truncate(summary);
      const name = this.nameOf(id);
      return {
        id,
        label: name,
        cells: [name, ...keys.map((f) => (rec[f.key] === undefined ? "—" : formatValue(rec[f.key]!, f))), cut],
        tips: [name === id ? `member "${id}" (its document has not been read yet)` : id, ...keys.map(() => undefined), summary.length > cut.length ? summary : undefined],
      };
    });
  }

  /** build the rows for `sweep` at `current` (the member on view, whose bundle is loaded and described by `memberInfo`) */
  build(sweep: Sweep | undefined, current: string, memberInfo: Info | undefined): void {
    this.sweep = sweep;
    if (!sweep) { this.host.style.display = "none"; return; }
    this.host.style.display = "";
    this.facets = sweep.facets(current);
    const record = sweep.record(current);
    const fmtOf = (key: string) => (v: RecordValue) => formatValue(v, this.facets.find((f) => f.key === key));
    const keys = this.varying();

    // the member row: picker (columns follow the sweep's varying keys) and ⓘ with the record table
    this.memberRow.querySelector("label")!.dataset.tip = `The member of the sweep on view (${sweep.memberIds.length} members). Click for a table of the members; scroll or use ↑/↓ while hovering to step; the lines below move along one key at a time. ?member=id`;
    const columns: PickerColumn[] = [{ title: "member", cls: "name" }, ...keys.map((f) => ({ title: f.name, cls: "dim" })), { title: "summary", cls: "summary" }];
    this.picker.setColumns(columns);
    this.picker.set(current, memberInfo?.name ?? current);
    const table: InfoTable = this.facets.filter((f) => record[f.key] !== undefined).map((f) => [f.name, fmtOf(f.key)(record[f.key]!)]);
    bindInfoIcon(this.info, memberInfo, table, memberInfo?.name ?? current);

    // one line per varying key
    this.lines.replaceChildren();
    for (const f of keys) {
      const row = document.createElement("div"); row.className = "prow keyline";
      const spacer = document.createElement("label"); spacer.textContent = "·"; // keeps the key column's width; hidden
      const inl = document.createElement("div"); inl.className = "inl";
      const klabel = document.createElement("span"); klabel.className = "klabel"; klabel.textContent = `${f.name}:`;
      klabel.dataset.tip = `${f.spec?.summary ?? `the sweep key "${f.key}"`} — ${f.kind}, ${f.values.filter((v) => v.members.length).length} values across the members. Click a value to switch to the member with it that changes the fewest other keys.`;
      const bar = document.createElement("div"); bar.className = "ch record"; bar.dataset.justify = "left";
      for (const v of f.values) {
        const seg = document.createElement("div"); seg.className = "seg"; seg.textContent = fmtOf(f.key)(v.value);
        const isCurrent = record[f.key] === v.value;
        const target = isCurrent ? undefined : sweep.nearest(current, f.key, v.value);
        seg.classList.toggle("on", isCurrent);
        seg.classList.toggle("disabled", !v.members.length);
        seg.classList.toggle("indirect", !isCurrent && !v.direct && v.members.length > 0);
        if (!v.members.length) seg.dataset.tip = `${f.name} = ${fmtOf(f.key)(v.value)}: no member of the sweep has this value`;
        else if (isCurrent) seg.dataset.tip = `${f.name} = ${fmtOf(f.key)(v.value)}: this member's value (${v.members.length} member${v.members.length === 1 ? "" : "s"} share it)`;
        else if (target) {
          // what else the switch changes — coordinates only; attributes (parameter count, test accuracy) follow the member
          const changes = changedKeys(record, sweep.record(target), f.key).filter(([k]) => !this.facets.find((x) => x.key === k)?.attribute).map(([k, val]) => `${this.facets.find((x) => x.key === k)?.name ?? k} → ${val === undefined ? "—" : fmtOf(k)(val)}`);
          seg.dataset.tip = changes.length ? `${f.name} → ${fmtOf(f.key)(v.value)} also changes ${changes.join(", ")} (no member differs only in ${f.name}); switches to "${this.nameOf(target)}"` : `${f.name} → ${fmtOf(f.key)(v.value)}: switches to "${this.nameOf(target)}", nothing else changes`;
          seg.addEventListener("click", () => this.onPick?.(target, [`${f.name} → ${fmtOf(f.key)(v.value)}`, ...changes].join(", ")));
        }
        bar.appendChild(seg);
      }
      inl.append(klabel, bar);
      row.append(spacer, inl);
      this.lines.appendChild(row);
    }
    installTooltips(this.lines);
  }
}

/** the keys on which `to` differs from `from` (present in either), except `except`; a key `to` lacks reads undefined */
function changedKeys(from: Record<string, RecordValue>, to: Record<string, RecordValue>, except: string): [string, RecordValue | undefined][] {
  const out: [string, RecordValue | undefined][] = [];
  for (const k of new Set([...Object.keys(from), ...Object.keys(to)])) {
    if (k === except) continue;
    if (from[k] !== to[k]) out.push([k, to[k]]);
  }
  return out;
}
