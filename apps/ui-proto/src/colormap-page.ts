// Colormap interval page: every .isl.cmap becomes an interval slider whose background is its colormap (data-map),
// stretched over the selected interval. One raster (x² − y²) is coloured through the "w1" bar's interval.
import { COLORMAPS, MAPS } from "../../viewer/src/colormap";
import { type CmapIntervalEl, makeCmapInterval, paramOf } from "./cmapInterval";
import { formatInterval } from "./format";
import { makeIntervalSlider } from "./interval";

type MapName = (typeof MAPS)[number];
const intervals: Record<string, CmapIntervalEl> = {};
const maps: Record<string, MapName> = {};

for (const el0 of document.querySelectorAll<HTMLElement>(".isl.cmap")) {
  const name = (el0.dataset.map as MapName) ?? "viridis";
  const el = makeCmapInterval(makeIntervalSlider(el0), COLORMAPS[name]);
  intervals[el.id] = el; maps[el.id] = name;
  const out = document.getElementById(`${el.id}v`);
  if (out) {
    const upd = () => {
      out.innerHTML = el.lo === null && el.hi === null ? `<span class="dim">none</span>` : formatInterval(el.lo, el.hi);
    };
    el.addEventListener("input", upd); el.addEventListener("change", upd); upd();
  }
}

/* click a name to cycle that bar's colormap (the bar's own clicks belong to the interval) */
for (const lab of document.querySelectorAll<HTMLElement>("[data-cycle]")) {
  const id = lab.dataset.cycle!;
  lab.addEventListener("click", () => {
    const next = MAPS[(MAPS.indexOf(maps[id]!) + 1) % MAPS.length]!;
    maps[id] = next; intervals[id]!.colormap = COLORMAPS[next]!;
    const text = [...lab.childNodes].find((n) => n.nodeType === Node.TEXT_NODE && n.textContent!.trim());
    if (text) text.textContent = next;
    if (id === "w1") drawField();
  });
}

/* the raster: f(x, y) = x² − y² over [-2, 2] × [-1, 1] (aspect of the canvas), coloured through w1 */
const canvas = document.getElementById("field") as HTMLCanvasElement | null;
function drawField(): void {
  if (!canvas) return;
  const bar = intervals.w1!, f = COLORMAPS[maps.w1!];
  const ctx = canvas.getContext("2d")!, W = canvas.width, H = canvas.height;
  const img = ctx.createImageData(W, H), d = img.data;
  for (let j = 0; j < H; j++) {
    const y = -1 + (2 * j) / (H - 1);
    for (let i = 0; i < W; i++) {
      const x = -2 + (4 * i) / (W - 1);
      const t = paramOf(bar, x * x - y * y), k = 4 * (j * W + i);
      if (t === null) { d[k + 3] = 0; continue; } // masked: not drawn
      const [r, g, b] = f(t);
      d[k] = r * 255; d[k + 1] = g * 255; d[k + 2] = b * 255; d[k + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}
if (intervals.w1) { intervals.w1.addEventListener("input", drawField); intervals.w1.addEventListener("change", drawField); drawField(); }

window.proto = { intervals, maps }; // declared in main.ts
