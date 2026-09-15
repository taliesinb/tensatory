# UI prototypes

Throwaway sandbox for new widgets before they move into `apps/viewer/src/widgets.ts`.
Nothing in here is imported by production code; it imports the viewer's existing
`makeSlider` (read-only) for side-by-side comparison.

```
pnpm --filter @tensatory/ui-proto dev     # http://127.0.0.1:5181/  (HMR)
pnpm --filter @tensatory/ui-proto build   # dist/index.html — ONE self-contained file, opens via file://
pnpm --filter @tensatory/ui-proto typecheck
```

* `src/interval.ts` — `makeIntervalSlider(el)`: `<div class="isl" data-min data-max data-step data-lo data-hi [data-nullable]>`.
  Exposes `.lo` / `.hi` / `.value` ("lo,hi" | null) / `.center` / `.width` / `setRange()`, fires `input` / `change`.
  The header comment lists every gesture.
* `src/main.ts` — demo wiring; `window.proto = { sliders, intervals, logged }` for scripted tests
  (dispatch `PointerEvent`s at fractions of the bar's rect; `setPointerCapture` is guarded so synthetic events work).
* `src/style.css` — the viewer's `.sl` rules verbatim + the new `.isl` rules.
