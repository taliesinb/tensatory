// Generate apps/ui-proto/icons.html: T_{μν} icon variants from KaTeX Math-Italic outlines (glyphs.json).
const fs = require("fs");
const g = JSON.parse(fs.readFileSync(__dirname + "/glyphs.json")).glyphs;
const rings = fs.readFileSync(__dirname + "/rings.svg", "utf8");
const paths = [...rings.matchAll(/<path d="([^"]+)" fill="([^"]+)"( fill-rule="evenodd")?\/>/g)].map((m) => ({ d: m[1], fill: m[2], evenodd: !!m[3] }));

const script = String.raw`
const G = __G__;
const RINGS = __RINGS__;
const Y = "#d2e436", O = "#ff781b", R = "#d2350e", BG = "#000", DARK = "#0b0d12", WHITE = "#e8ecf5";
// glyph paths are in a 1000-unit em, y down, baseline at 0
function formula({ x, y, size, color, subColor, drop = 0.25, subScale = 0.7, gap = 0, stroke = 0 }) {
  const s = size / 1000, ss = s * subScale;
  const subX = x + (G.T.adv + gap * 1000) * s, subY = y + drop * size;
  const st = (c) => stroke ? ' fill="none" stroke="' + c + '" stroke-width="' + stroke + '" stroke-linejoin="round"' : ' fill="' + c + '"';
  return '<g transform="translate(' + x + ',' + y + ') scale(' + s + ')"><path d="' + G.T.d + '"' + st(color) + '/></g>' +
    '<g transform="translate(' + subX + ',' + subY + ') scale(' + ss + ')"><path d="' + G.mu.d + '"' + st(subColor) + '/><path transform="translate(' + G.mu.adv + ',0)" d="' + G.nu.d + '"' + st(subColor) + '/></g>';
}
const fw = (gap, subScale) => (G.T.adv + gap * 1000 + (G.mu.adv + G.nu.bbox.x2) * subScale) / 1000;
function centered(opts) {
  const { size, gap = 0, subScale = 0.7, drop = 0.25 } = opts;
  const w = fw(gap, subScale) * size;
  const top = -0.677 * size, bottom = (drop + 0.216 * subScale) * size;
  return formula({ x: (100 - w) / 2 - (G.T.bbox.x1 / 1000) * size * 0.5, y: 50 - (top + bottom) / 2, ...opts });
}
function ringsSvg(alpha, scale, cx = 50, cy = 50) {
  return '<g opacity="' + alpha + '" transform="translate(' + cx + ',' + cy + ') scale(' + scale + ') translate(-50,-50)">' +
    RINGS.map((p) => '<path d="' + p.d + '" fill="' + p.fill + '"' + (p.evenodd ? ' fill-rule="evenodd"' : "") + "/>").join("") + "</g>";
}
const tile = (bg, r) => '<rect width="100" height="100" rx="' + r + '" fill="' + bg + '"/>';
const VIRIDIS = '<defs><linearGradient id="vg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fde725"/><stop offset="0.5" stop-color="#21918c"/><stop offset="1" stop-color="#440154"/></linearGradient></defs>';
const variants = [
  ["A", "plain", "Formula alone, the rings' yellow on black (browsers mask favicons themselves, so no rounding).", () => tile(BG, 0) + centered({ size: 62, color: Y, subColor: Y })],
  ["B", "two-tone", "T in yellow, μν in orange — the two hot colours of the current rings.", () => tile(BG, 0) + centered({ size: 62, color: Y, subColor: O })],
  ["C", "white on dark", "Neutral: the viewer's #0b0d12 background, white glyphs (matches the UI chrome).", () => tile(DARK, 0) + centered({ size: 62, color: WHITE, subColor: WHITE })],
  ["D", "over faint rings", "Formula over the existing isoline rings at 22% — keeps the lineage of the old icon.", () => tile(BG, 0) + ringsSvg(0.22, 1.25) + centered({ size: 60, color: Y, subColor: Y })],
  ["E", "rings behind, two-tone", "Same, rings at 30%, formula two-tone.", () => tile(BG, 0) + ringsSvg(0.3, 1.3) + centered({ size: 58, color: Y, subColor: O })],
  ["F", "rings as a corner motif", "Rings small in the lower right (the loss-landscape hint), formula upper-left.", () => tile(BG, 0) + ringsSvg(0.9, 0.42, 78, 78) + formula({ x: 6, y: 58, size: 66, color: Y, subColor: Y })],
  ["G", "tight subscript", "Subscript closer and lower (drop 0.32, gap −0.04): more like printed T_{μν}.", () => tile(BG, 0) + centered({ size: 64, color: Y, subColor: Y, drop: 0.32, gap: -0.04 })],
  ["H", "large, cropped", "Formula fills the tile (size 76) — reads at 16 px; the T's serif nearly touches the edge.", () => tile(BG, 0) + centered({ size: 76, color: Y, subColor: Y, gap: -0.03 })],
  ["I", "outline", "Hollow glyphs (stroke only), like isolines.", () => tile(BG, 0) + centered({ size: 62, color: Y, subColor: Y, stroke: 28 })],
  ["J", "red T", "Reversed palette: T in the innermost ring's red, μν yellow.", () => tile(BG, 0) + centered({ size: 62, color: R, subColor: Y })],
  ["K", "squircle tile", "Rounded tile in the ring yellow, black glyphs — for contexts that do not mask (Dock, home screen).", () => tile(Y, 22) + centered({ size: 62, color: BG, subColor: BG })],
  ["L", "viridis glyphs", "Glyphs filled with the viewer's colormap ramp (yellow → teal → purple), top to bottom.", () => tile(DARK, 0) + VIRIDIS + centered({ size: 62, color: "url(#vg)", subColor: "url(#vg)" })],
  ["M", "black on white · KaTeX spacing", "Black on white, subscript where KaTeX puts it (gap 0, drop 0.25).", () => tile("#fff", 0) + centered({ size: 64, color: BG, subColor: BG })],
  ["N", "snug 1", "Subscript tucked under the T's arm: gap −0.10 em, drop 0.28.", () => tile("#fff", 0) + centered({ size: 66, color: BG, subColor: BG, drop: 0.28, gap: -0.10 })],
  ["O", "snug 2", "Closer still: gap −0.16, drop 0.30 (the μ's left edge under the arm's serif).", () => tile("#fff", 0) + centered({ size: 68, color: BG, subColor: BG, drop: 0.30, gap: -0.16 })],
  ["P", "snug 3", "Tightest: gap −0.22, drop 0.32 — the μ starts right of the T's stem.", () => tile("#fff", 0) + centered({ size: 70, color: BG, subColor: BG, drop: 0.32, gap: -0.22 })],
  ["Q", "snug 2, larger subscript", "gap −0.16, drop 0.30, subscript at 0.78× instead of 0.7×.", () => tile("#fff", 0) + centered({ size: 66, color: BG, subColor: BG, drop: 0.30, gap: -0.16, subScale: 0.78 })],
  ["R", "snug 2, filling the tile", "Same as O at size 78: reads at 16 px.", () => tile("#fff", 0) + centered({ size: 78, color: BG, subColor: BG, drop: 0.30, gap: -0.16 })],
  ["S", "snug 2, rounded tile", "O on a white squircle with a hairline border, for contexts that do not mask.", () => tile("#fff", 22) + '<rect x="0.75" y="0.75" width="98.5" height="98.5" rx="21.5" fill="none" stroke="#c8ccd6" stroke-width="1.5"/>' + centered({ size: 66, color: BG, subColor: BG, drop: 0.30, gap: -0.16 })],
];
const grid = document.getElementById("grid");
window.__icons = {};
for (const [id, name, desc, make] of variants) {
  const inner = make();
  const svg = (px) => '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="' + px + '" height="' + px + '">' + inner + "</svg>";
  const card = document.createElement("div"); card.className = "card";
  card.innerHTML = "<h2>" + id + " · " + name + "</h2><p>" + desc + "</p><div class=sizes>" + [180, 64, 32, 16].map((px) => "<div>" + svg(px) + '<div class=lbl>' + px + "</div></div>").join("") + "</div>";
  grid.appendChild(card);
  window.__icons[id] = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">' + inner + "</svg>";
}
`.replace("__G__", JSON.stringify(g)).replace("__RINGS__", JSON.stringify(paths));

const html = `<!doctype html>
<html><head><meta charset="utf-8"><title>Tensatory icon prototypes: T_{μν}</title>
<style>
body{background:#e9ebf0;color:#2a3244;font:13px -apple-system,system-ui,sans-serif;margin:0;padding:24px}
h1{font-size:16px;font-weight:600;margin:0 0 16px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:24px}
.card{background:#fff;border:1px solid #cfd4de;border-radius:10px;padding:14px}
.card h2{font-size:13px;font-weight:600;margin:0 0 4px;color:#111}.card p{margin:0 0 10px;color:#6b748a;font-size:12px;min-height:32px}
.sizes{display:flex;gap:12px;align-items:flex-end}.sizes svg{display:block}
.lbl{font-size:10px;color:#8b95ab;text-align:center;margin-top:4px}
</style></head><body>
<h1>Tensatory icon prototypes — T<sub>μν</sub> in KaTeX Math Italic (outlines baked into the SVG, no font needed)</h1>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js" onload="katex.render(String.fromCharCode(84,95,123,92,109,117,92,110,117,125), document.getElementById('ref'), { displayMode: true })"></script>
<div class="card" style="display:inline-block;margin-bottom:24px"><h2>reference · KaTeX itself</h2><p>The formula as KaTeX renders it (webfont, CDN), for comparing the baked spacing.</p><div id="ref" style="font-size:96px;color:#000;background:#fff;padding:8px 24px;display:inline-block"></div></div>
<div class="grid" id="grid"></div>
<script>${script}</script></body></html>`;
fs.writeFileSync(__dirname + "/../../apps/ui-proto/icons.html", html);
console.log("written", html.length);
