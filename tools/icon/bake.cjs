// Bake variant S into apps/viewer/public/icons/icon.svg. Same geometry as icons.html: white squircle (rx 22), hairline
// border, T_{μν} in KaTeX Math Italic, subscript 0.7×, gap −0.16 em, drop 0.30, size 66.
const fs = require("fs");
const G = JSON.parse(fs.readFileSync(__dirname + "/glyphs.json")).glyphs;
const size = 66, gap = -0.16, drop = 0.30, subScale = 0.7;
const s = size / 1000, ss = s * subScale;
const w = (G.T.adv + gap * 1000 + (G.mu.adv + G.nu.bbox.x2) * subScale) / 1000 * size;
const top = -0.677 * size, bottom = (drop + 0.216 * subScale) * size;
const x = (100 - w) / 2 - (G.T.bbox.x1 / 1000) * size * 0.5, y = 50 - (top + bottom) / 2;
const subX = x + (G.T.adv + gap * 1000) * s, subY = y + drop * size;
const r = (v) => Math.round(v * 1000) / 1000;
const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
<!-- Tensatory: T_{μν} in KaTeX Math Italic (glyph outlines baked; the font is not needed) on a white squircle -->
<rect width="100" height="100" rx="22" fill="#fff"/>
<rect x="0.75" y="0.75" width="98.5" height="98.5" rx="21.5" fill="none" stroke="#c8ccd6" stroke-width="1.5"/>
<g fill="#000">
<path transform="translate(${r(x)},${r(y)}) scale(${r(s)})" d="${G.T.d}"/>
<path transform="translate(${r(subX)},${r(subY)}) scale(${r(ss)})" d="${G.mu.d}"/>
<path transform="translate(${r(subX + G.mu.adv * ss)},${r(subY)}) scale(${r(ss)})" d="${G.nu.d}"/>
</g>
</svg>
`;
fs.writeFileSync(__dirname + "/../../apps/viewer/public/icons/icon.svg", svg);
console.log("icon.svg", svg.length, "bytes");
