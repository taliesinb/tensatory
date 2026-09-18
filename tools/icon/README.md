# Icon

`T_{μν}` set in KaTeX Math Italic on a white squircle (variant **S** of the
prototypes). The glyph outlines are baked into the SVG, so no font is needed.

* `glyphs.json` — outlines of *T*, *μ*, *ν* extracted from
  `KaTeX_Math-Italic.ttf` (katex 0.16.11) with opentype.js:
  `getPath(0, 0, 1000).toPathData(2)`, plus advance widths and bounding boxes.
* `bake.cjs` — writes `apps/viewer/public/icons/icon.svg` (subscript 0.7×,
  gap −0.16 em under the T's arm, drop 0.30 em, size 66 in a 100 box). Then:
  `cd apps/viewer/public/icons && for n in 180 192 512 1024; do rsvg-convert -w $n -h $n icon.svg -o icon-$n.png; done`
* `prototypes.cjs` — regenerates `apps/ui-proto/icons.html`, the page the
  variants were chosen from (`rings.svg` is the previous isoline-rings icon,
  used as a backdrop in some variants).
