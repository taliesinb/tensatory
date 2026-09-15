// Demo page wiring: instantiate every .sl (the viewer's slider, imported unchanged) and .isl (the prototype),
// show their values, log events, and expose everything on window.proto for scripted testing.
import { makeSlider } from "../../viewer/src/widgets";
import { makeIntervalSlider, type IntervalEl } from "./interval";
import { formatInterval } from "./format";

const $ = (id: string) => document.getElementById(id)!;

const sliders: Record<string, HTMLElement> = {};
for (const el of document.querySelectorAll<HTMLElement>(".sl")) sliders[el.id] = makeSlider(el);
const intervals: Record<string, IntervalEl> = {};
for (const el of document.querySelectorAll<HTMLElement>(".isl")) intervals[el.id] = makeIntervalSlider(el);

/* value readouts */
const readout = (el: IntervalEl) => {
  const out = document.getElementById(`${el.id}v`); if (!out) return;
  const upd = () => {
    out.innerHTML = el.lo === null && el.hi === null ? `<span class="dim">none</span>` : formatInterval(el.lo, el.hi);
  };
  el.addEventListener("input", upd); el.addEventListener("change", upd); upd();
};
for (const el of Object.values(intervals)) readout(el);
{
  const el = sliders.plain as HTMLElement & { value: string | null }, out = $("plainv");
  const upd = () => (out.textContent = el.value ?? "unset");
  el.addEventListener("input", upd); el.addEventListener("change", upd); upd();
}

/* event log for the programmatic panel */
const log = $("log"), lines: string[] = [];
export const logged: { type: string; value: string | null; t: number }[] = [];
const note = (s: string) => { lines.push(s); if (lines.length > 200) lines.shift(); log.textContent = lines.join("\n"); log.scrollTop = log.scrollHeight; };
for (const el of Object.values(intervals)) {
  for (const t of ["input", "change"]) el.addEventListener(t, () => { logged.push({ type: t, value: el.value, t: performance.now() }); note(`${el.id.padEnd(9)} ${t.padEnd(6)} ${el.value ?? "null"}`); });
}
const prog = intervals.prog!;
$("setA").onclick = () => { prog.value = "0.4,0.9"; note("set value = 0.4,0.9"); prog.dispatchEvent(new Event("change")); };
$("setB").onclick = () => { prog.lo = 0.2; note("set lo = 0.2"); prog.dispatchEvent(new Event("change")); };
$("setC").onclick = () => { prog.setRange(-5, 5, 0.5); note("setRange(-5, 5, .5)"); prog.dispatchEvent(new Event("change")); };
$("setD").onclick = () => { prog.setRange(0, 1, 0.001); note("setRange(0, 1, .001)"); prog.dispatchEvent(new Event("change")); };

declare global { interface Window { proto: unknown } } // per-page bag of widgets for scripted tests
window.proto = { sliders, intervals, logged };
