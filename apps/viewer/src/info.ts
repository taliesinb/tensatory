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

/** show an Info's details (else its summary) in the info modal */
export function showInfo(i: Info): void {
  $("infoTitle").textContent = i.name;
  $("infoText").textContent = clickText(i);
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

/** a <span class="info"> (HTML ⓘ) shows `info`, or hides itself when there is none; the tooltip text lives in data-tip */
export function bindInfoIcon(el: HTMLElement, info: Info | undefined): void {
  el.style.display = info ? "" : "none";
  el.dataset.tip = info ? hoverText(info) : "";
  el.onclick = info ? () => showInfo(info) : null;
}
