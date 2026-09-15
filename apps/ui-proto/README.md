# UI prototypes

Throwaway sandbox for new widgets before they move into `apps/viewer/src/widgets.ts`.
Nothing in here is imported by production code; it imports the viewer's existing
`makeSlider` (read-only) for side-by-side comparison.

```
pnpm --filter @tensatory/ui-proto dev     # http://127.0.0.1:5181/ and /colormap.html  (HMR)
pnpm --filter @tensatory/ui-proto build   # dist/index.html + dist/colormap.html — each ONE self-contained file (file:// works)
pnpm --filter @tensatory/ui-proto typecheck
```

* `src/interval.ts` — `makeIntervalSlider(el)`: `<div class="isl" data-min data-max data-step data-lo data-hi>`
  (`"null"` on either end for a half / empty interval). Exposes `.lo` / `.hi` (each `number | null`),
  `.value` (`"lo,hi"` with `null` for a missing end; `null` when both are missing), `.center` / `.width`
  (full intervals only), `setRange()`; fires `input` / `change`. The header comment lists every gesture
  for the full / half / none kinds.
* `src/cmapInterval.ts` — colormap interval selection on top of the same widget (`.isl.cmap`, drawn as a box):
  `makeCmapInterval(el, colormap)` adds the modes that are part of the selection's value — `span`
  (`stretch` | `full`), `low` / `high` (`clip` | `mask`) — flips them on the widget's `modetoggle` clicks,
  repaints the bar (`cmapBackground`: hatch layers over masked sides + `colourGradient`) and exposes
  `paramOf` (value → colormap parameter, `null` where masked). Page: `colormap.html` + `src/colormap-page.ts`.
* `src/format.ts` — `formatReal` (copied from core) and `formatInterval`.
* `src/main.ts` — interval-slider demo wiring (`index.html`); `window.proto = { sliders, intervals, logged }` for scripted tests
  (dispatch `PointerEvent`s at fractions of the bar's rect; `setPointerCapture` is guarded so synthetic events work).
* `src/style.css` — the viewer's `.sl` rules verbatim + the new `.isl` rules.
