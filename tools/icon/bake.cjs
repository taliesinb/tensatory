// Bake the icon: T_{μν} in KaTeX Math Italic on a white squircle (variant S of apps/ui-proto/icons.html), at several
// size classes. The favicon SVG cannot know its render size, so the 16 and 32 px PNGs are their own bakes with the
// subscript raised (smaller bounding box, more ink) and the formula filling more of the tile.
const fs = require("fs");
const G = JSON.parse(fs.readFileSync(__dirname + "/glyphs.json")).glyphs;

function svgFor({ size, gap, drop, subScale = 0.7, border = true }) {
  const s = size / 1000, ss = s * subScale;
  const w = (G.T.adv + gap * 1000 + (G.mu.adv + G.nu.bbox.x2) * subScale) / 1000 * size;
  const top = -0.677 * size, bottom = (drop + 0.216 * subScale) * size;
  const x = (100 - w) / 2 - (G.T.bbox.x1 / 1000) * size * 0.5, y = 50 - (top + bottom) / 2;
  const subX = x + (G.T.adv + gap * 1000) * s, subY = y + drop * size;
  const r = (v) => Math.round(v * 1000) / 1000;
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
<!-- Tensatory: T_{μν} in KaTeX Math Italic (glyph outlines baked; the font is not needed) on a white squircle -->
<rect width="100" height="100" rx="22" fill="#fff"/>
${border ? '<rect x="0.75" y="0.75" width="98.5" height="98.5" rx="21.5" fill="none" stroke="#c8ccd6" stroke-width="1.5"/>\n' : ""}<g fill="#000">
<path transform="translate(${r(x)},${r(y)}) scale(${r(s)})" d="${G.T.d}"/>
<path transform="translate(${r(subX)},${r(subY)}) scale(${r(ss)})" d="${G.mu.d}"/>
<path transform="translate(${r(subX + G.mu.adv * ss)},${r(subY)}) scale(${r(ss)})" d="${G.nu.d}"/>
</g>
</svg>
`;
}

// Optical sizing, like kerning: the smaller the icon, the higher the subscript sits (a smaller bounding box, more
// ink per pixel) and the more of the tile the formula fills. Interpolated in log2(px) between anchors; at ≥ 180 px
// the typeset proportions of variant S.
const ANCHORS = [
  // px,  size, gap,   drop, subScale, border
  [16, 76, -0.18, 0.08, 0.74, false],
  [32, 74, -0.16, 0.16, 0.72, true],
  [64, 70, -0.16, 0.23, 0.71, true],
  [180, 66, -0.16, 0.30, 0.70, true],
];
function paramsFor(px) {
  const t = Math.log2(Math.min(180, Math.max(16, px)));
  let i = 0; while (i < ANCHORS.length - 2 && t > Math.log2(ANCHORS[i + 1][0])) i++;
  const [a, b] = [ANCHORS[i], ANCHORS[i + 1]];
  const u = (t - Math.log2(a[0])) / (Math.log2(b[0]) - Math.log2(a[0]));
  const mix = (k) => a[k] + (b[k] - a[k]) * u;
  return { size: mix(1), gap: mix(2), drop: mix(3), subScale: mix(4), border: px >= 24 };
}
const out = __dirname + "/../../apps/viewer/public/icons/";
fs.writeFileSync(out + "icon.svg", svgFor(paramsFor(180)));
const { execSync } = require("child_process");
for (const px of [16, 32, 48, 64, 180, 192, 512, 1024]) {
  const svg = svgFor(paramsFor(px));
  const tmp = `/tmp/icon-${px}.svg`;
  fs.writeFileSync(tmp, svg);
  execSync(`rsvg-convert -w ${px} -h ${px} ${tmp} -o ${out}icon-${px}.png`);
  const p = paramsFor(px);
  console.log(`icon-${px}.png  size ${p.size.toFixed(1)} drop ${p.drop.toFixed(3)} sub ${p.subScale.toFixed(2)}`);
}
