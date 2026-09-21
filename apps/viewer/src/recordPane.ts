// The RECORD ROWS of the bundle panel (notes/sweeps.md §2): when the loaded document is a sweep, the rows between the
// `bundle` and `space` pickers show which member is on view and let the user move through the sweep along its keys.
//
//   member    convnet · pca                       ⓘ    the member's name and its own summary / details
//   dataset   [mnist] fmnist                           one flipper per VARYING key; blue = this member's value
//   model     [mlp]  convnet                           tinted = a DIRECT switch (only this key changes);
//   dirs      [pca]  random                            dim = the value exists, but reaching it changes other keys too
//   record    seed 0 · test acc 0.976                  the keys that do not vary, as text
//
// Nothing here knows what a key means: core's `facets` discovers the values and their reachability from the records,
// `nearestMember` picks the member a click leads to (the fewest other keys changed). Keys the current member's record
// lacks are simply not shown (a convnet has no `num_layers`).

import { Codomain, formatReal, type Facet, type Info, type Sweep } from "@tensatory/core";
import type { RecordValue } from "@tensatory/schema";
import { bindInfoIcon } from "./info";
import { installTooltips } from "./widgets";

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
  constructor(private readonly host: HTMLElement) {}

  /** build the rows for `sweep` at `current`; a click hands the member to switch to `onPick`, with what the switch changes */
  build(sweep: Sweep | undefined, current: string, memberInfo: Info | undefined, onPick: (id: string, why: string) => void): void {
    this.host.replaceChildren();
    if (!sweep || !sweep.hasFacets) { this.host.style.display = "none"; return; }
    this.host.style.display = "";
    const facets = sweep.facets(current);
    const record = sweep.record(current);
    const fmtOf = (key: string) => (v: RecordValue) => formatValue(v, facets.find((f) => f.key === key));

    // the member: its name (the bundle's) and ⓘ
    {
      const row = document.createElement("div"); row.className = "prow";
      const label = document.createElement("label"); label.textContent = "member";
      label.dataset.tip = `The member of the sweep on view (${sweep.memberIds.length} members). The rows below move through the sweep along its keys; ?member=id`;
      const name = document.createElement("span"); name.className = "only"; name.textContent = memberInfo?.name ?? current;
      const info = document.createElement("span"); info.className = "info"; info.textContent = "i"; info.dataset.tip = "";
      bindInfoIcon(info, memberInfo);
      row.append(label, name, info);
      this.host.appendChild(row);
    }

    for (const f of facets) {
      if (!f.varying || f.attribute) continue;
      const row = document.createElement("div"); row.className = "prow";
      const label = document.createElement("label"); label.textContent = f.name;
      label.dataset.tip = `${f.spec?.summary ?? `the sweep key "${f.key}"`} — ${f.kind}, ${f.values.filter((v) => v.members.length).length} values across the members. Click a value to switch to the member with it that changes the fewest other keys.`;
      const bar = document.createElement("div"); bar.className = "ch multi record"; bar.dataset.justify = "left";
      for (const v of f.values) {
        const seg = document.createElement("div"); seg.className = "seg"; seg.textContent = fmtOf(f.key)(v.value);
        const isCurrent = record[f.key] === v.value;
        const target = isCurrent ? undefined : sweep.nearest(current, f.key, v.value);
        seg.classList.toggle("on", isCurrent);
        seg.classList.toggle("disabled", !v.members.length);
        seg.classList.toggle("direct", !isCurrent && v.direct);
        if (!v.members.length) seg.dataset.tip = `${f.name} = ${fmtOf(f.key)(v.value)}: no member of the sweep has this value`;
        else if (isCurrent) seg.dataset.tip = `${f.name} = ${fmtOf(f.key)(v.value)}: this member's value (${v.members.length} member${v.members.length === 1 ? "" : "s"} share it)`;
        else if (target) {
          // what else the switch changes — coordinates only; attributes (parameter count, test accuracy) follow the member and are shown in the record row
          const changes = changedKeys(record, sweep.record(target), f.key).filter(([k]) => !facets.find((x) => x.key === k)?.attribute).map(([k, val]) => `${facets.find((x) => x.key === k)?.name ?? k} → ${val === undefined ? "—" : fmtOf(k)(val)}`);
          seg.dataset.tip = changes.length ? `${f.name} → ${fmtOf(f.key)(v.value)} also changes ${changes.join(", ")} (no member differs only in ${f.name}); switches to "${target}"` : `${f.name} → ${fmtOf(f.key)(v.value)}: switches to "${target}", nothing else changes`;
          seg.addEventListener("click", () => onPick(target, [`${f.name} → ${fmtOf(f.key)(v.value)}`, ...changes].join(", ")));
        }
        bar.appendChild(seg);
      }
      row.append(label, bar);
      this.host.appendChild(row);
    }

    // the rest of the record: keys that do not vary, and the member's attributes (measurements) whether they vary or not
    const fixed = facets.filter((f) => (!f.varying || f.attribute) && record[f.key] !== undefined);
    if (fixed.length) {
      const row = document.createElement("div"); row.className = "prow";
      const label = document.createElement("label"); label.textContent = "record";
      label.dataset.tip = "The rest of this member's record: keys that are the same on every member, and the member's own measurements (test accuracy, parameter count, …) — attributes of the member, not coordinates of the sweep.";
      const val = document.createElement("div"); val.className = "pval";
      val.textContent = fixed.map((f) => `${f.name} ${fmtOf(f.key)(record[f.key]!)}`).join(" · ");
      row.append(label, val);
      this.host.appendChild(row);
    }
    installTooltips(this.host);
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
