# Icon

`T_{μν}` set in KaTeX Math Italic on a white squircle (variant **S** of the
prototypes). The glyph outlines are baked into the SVG, so no font is needed.

* `glyphs.json` — outlines of *T*, *μ*, *ν* extracted from
  `KaTeX_Math-Italic.ttf` (katex 0.16.11) with opentype.js:
  `getPath(0, 0, 1000).toPathData(2)`, plus advance widths and bounding boxes.
* `bake.cjs` — writes `apps/viewer/public/icons/icon.svg` and every PNG (16,
  32, 48, 64, 180, 192, 512, 1024; needs `rsvg-convert`). The subscript is
  OPTICALLY SIZED, like kerning: at ≥ 180 px the typeset proportions
  (subscript 0.7×, gap −0.16 em under the T's arm, drop 0.30 em, size 66 in a
  100 box); as the icon shrinks the subscript rises and the formula grows
  (16 px: drop 0.08, size 76, no border), interpolated in log₂ px between
  anchors. `index.html` lists the 16 / 32 / 48 / 64 PNGs by size so browsers
  pick the matching bake; the SVG is the `any` fallback.
* `prototypes.cjs` — regenerates `apps/ui-proto/icons.html`, the page the
  variants were chosen from (`rings.svg` is the previous isoline-rings icon,
  used as a backdrop in some variants).
