// Summaries and details of bundles, spaces and fields (core `Info`: name + optional one-line `summary` + optional
// `details` of any length), and the ⓘ icon that exposes them:
//   * hovering an alternative in a <select> shows the summary, else the first line of the details (option `title`);
//   * hovering the ⓘ shows the summary, else the details (the shared [data-tip] tooltip: 0.25 s, cursor: help);
//   * clicking the ⓘ opens the details, else the summary, in a modal titled with the name.
// The icon is only shown when there is a summary or details.

import type { Info } from "@tensatory/core";

const $ = (id: string) => document.getElementById(id)!;

/** the first non-empty line */
export const firstLine = (s: string): string => s.split("\n").map((l) => l.trim()).find((l) => l.length) ?? "";
/** for a dropdown alternative: the summary, else the first line of the details */
export const optionText = (i: Info | undefined): string => (i ? i.summary ?? firstLine(i.details!) : "");
/** for hovering the ⓘ: the summary, else the details */
export const hoverText = (i: Info): string => i.summary ?? i.details!;
/** for clicking the ⓘ: the details, else the summary */
export const clickText = (i: Info): string => i.details ?? i.summary!;

/** a key / value table shown under the text in the modal (a sweep member's record) */
export type InfoTable = [string, string][];

/** a themed two-column table of key / value pairs (text only); shared by the tooltip and the info modal */
export function kvTable(rows: InfoTable): HTMLTableElement {
  const table = document.createElement("table"); table.className = "kvtable";
  for (const [k, v] of rows) {
    const tr = document.createElement("tr");
    const kd = document.createElement("td"); kd.className = "k"; kd.textContent = k;
    const vd = document.createElement("td"); vd.className = "v"; vd.textContent = v;
    tr.append(kd, vd); table.appendChild(tr);
  }
  return table;
}

/** show an Info's details (else its summary) in the info modal, with the table (when given) as a section below */
export function showInfo(i: Info, table?: InfoTable): void {
  $("infoTitle").textContent = i.name;
  const text = i.details ?? i.summary ?? "";
  $("infoText").textContent = text;
  $("infoText").style.display = text ? "" : "none";
  const t = $("infoTable");
  t.replaceChildren();
  if (table?.length) { t.appendChild(kvTable(table)); t.style.display = ""; } else t.style.display = "none";
  $("infoModal").classList.add("open");
}
export function closeInfo(): void { $("infoModal").classList.remove("open"); }

/** wire the modal's close affordances once: ✕, a click on the backdrop, Escape */
export function installInfoModal(): void {
  const modal = $("infoModal");
  $("infoClose").addEventListener("click", closeInfo);
  modal.addEventListener("pointerdown", (e) => { if (e.target === modal) closeInfo(); });
  window.addEventListener("keydown", (e) => { if (e.key === "Escape" && modal.classList.contains("open")) closeInfo(); });
}

/**
 * a <span class="info"> (HTML ⓘ) shows `info` (and, in the modal, `table`), or hides itself when there is neither;
 * the tooltip text lives in data-tip. With a table but no summary / details the icon still opens the modal.
 */
export function bindInfoIcon(el: HTMLElement, info: Info | undefined, table?: InfoTable, name?: string): void {
  const show = !!info || !!table?.length;
  el.style.display = show ? "" : "none";
  el.dataset.tip = info ? hoverText(info) : table?.length ? `${name ?? ""}: click for its record`.replace(/^: /, "") : "";
  el.onclick = show ? () => showInfo(info ?? { name: name ?? "" }, table) : null;
}
