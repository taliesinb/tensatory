// Interval slider prototype — a two-ended sibling of the viewer's compact slider (apps/viewer/src/widgets.ts).
//
//   <div class="isl" data-min="0" data-max="1" data-step="0.001" data-lo="0.2" data-hi="0.6" [data-nullable]></div>
//
// The root element owns its state (lo, hi) and creates its own presentation children, exactly like `.sl`
// creates `.fill` / `.was`:
//   .band  the interval itself (translucent), .lo / .hi  3px edge handles, .mid  a thin centre marker,
//   .was   dotted ghosts of the committed lo / hi while a drag is in flight.
// The centre is derived ((lo + hi) / 2), never stored: dragging the band translates both ends (width kept,
// clipped at the bar's ends); dragging a handle moves that end (clamped so lo <= hi).
//
// Interaction (mirrors the compact slider wherever it has an analogue):
//   press on a handle  -> drag that end          press inside the band -> drag the whole interval
//   press outside      -> centre jumps there, then drags (like the slider's click-to-jump)
//   alt-press anywhere -> symmetric resize about the current centre (also the way out when lo == hi and
//                         the two handles overlap: a plain press there drags the band)
//   shift-hover        -> preview: centre follows the pointer until shift is released
//   wheel              -> translate (1000-position clamped "document"); alt-wheel -> resize
//   ← →                -> nudge centre by 10 % of the range; alt-← / alt-→ -> narrow / widen by 10 %
//   Escape             -> cancel the drag;  data-nullable: quick click (no drag, <= 500 ms) on the band or
//                         Backspace unsets (.lo === .hi === .value === null); clicking an unset bar re-creates
//                         the last interval centred where you clicked.
// Events: 'input' while the shown interval moves (drag / preview), 'change' when it is committed.

export interface IntervalEl extends HTMLElement {
  lo: number | null;
  hi: number | null;
  /** "lo,hi" or null; the setter also accepts [lo, hi] */
  value: string | null;
  readonly center: number | null;
  readonly width: number | null;
  setRange(min: number, max: number, step?: number): void;
}

type Mode = "lo" | "hi" | "band" | "scale";

export function makeIntervalSlider(el0: HTMLElement): IntervalEl {
  const el = el0 as IntervalEl;
  let min = +el.dataset.min!, max = +el.dataset.max!, step = +(el.dataset.step ?? 0);
  const nullable = el.dataset.nullable !== undefined;
  const span = () => max - min;
  const quant = (v: number) => { v = Math.min(max, Math.max(min, v)); if (step) v = min + Math.round((v - min) / step) * step; return +v.toFixed(6); };

  // stored = committed, shown = what is painted (differs during drags / previews)
  // data-lo / data-hi default to the ends of the range; "null" on either (with data-nullable) starts unset,
  // remembering the other end's value (or the full range) as the interval a click will bring back
  const num = (s: string | undefined, dflt: number) => { const v = s === undefined || s === "null" ? NaN : +s; return Number.isFinite(v) ? v : dflt; };
  let sLo = quant(num(el.dataset.lo, min)), sHi = quant(num(el.dataset.hi, max));
  if (sHi < sLo) [sLo, sHi] = [sHi, sLo];
  let lo = sLo, hi = sHi;
  let isNull = nullable && (el.dataset.lo === "null" || el.dataset.hi === "null");
  let dragging = false, moved = false, mode: Mode = "band";
  let downX = 0, downT = 0, downOff = 0, downW = 0, downC = 0, downOnThing = false, downWasNull = false;
  let settled = false, over = false, previewing = false, lastEv: PointerEvent | null = null;

  const mk = (cls: string) => { const d = document.createElement("div"); d.className = cls; el.appendChild(d); return d; };
  const wasLo = mk("was"), wasHi = mk("was"), band = mk("band"), mid = mk("mid"), hLo = mk("lo"), hHi = mk("hi");

  const pct = (v: number) => `${((v - min) / span()) * 100}%`;
  const paint = () => {
    band.style.left = pct(lo); band.style.width = pct(min + hi - lo);
    hLo.style.left = pct(lo); hHi.style.left = pct(hi); mid.style.left = pct((lo + hi) / 2);
    wasLo.style.left = pct(sLo); wasHi.style.left = pct(sHi);
    const ghost = !isNull && (lo !== sLo || hi !== sHi);
    wasLo.style.display = ghost && sLo !== lo ? "block" : "none";
    wasHi.style.display = ghost && sHi !== hi ? "block" : "none";
    el.classList.toggle("null", isNull);
    el.classList.toggle("narrow", hi - lo === 0);
  };
  const fire = (t: string) => el.dispatchEvent(new Event(t));
  const rect = () => el.getBoundingClientRect();
  const atX = (clientX: number) => { const r = rect(); return min + ((clientX - r.left) / r.width) * span(); }; // unquantized
  const xOf = (v: number) => { const r = rect(); return r.left + ((v - min) / span()) * r.width; };

  /** show [l, h] (quantized, ordered, clamped) and fire input if anything changed */
  const show = (l: number, h: number) => {
    l = quant(l); h = quant(h); if (h < l) [l, h] = [h, l];
    if (l !== lo || h !== hi || isNull) { lo = l; hi = h; isNull = false; paint(); fire("input"); }
  };
  /** translate so the centre is at c, width w, clipped at the ends */
  const showCentred = (c: number, w: number) => { let l = c - w / 2; l = Math.max(min, Math.min(max - w, l)); show(l, l + w); };
  /** symmetric resize about c to half-width hw */
  const showScaled = (c: number, hw: number) => { hw = Math.max(0, Math.min(hw, c - min, max - c)); show(c - hw, c + hw); };

  /** what is under the pointer: a handle, the band, or nothing */
  const hit = (e: PointerEvent): "lo" | "hi" | "band" | "out" => {
    if (isNull) return "out";
    const x = e.clientX, xl = xOf(lo), xh = xOf(hi);
    const nearLo = Math.abs(x - xl) <= 4, nearHi = Math.abs(x - xh) <= 4;
    if (nearLo && nearHi) return "band"; // handles overlap: the interval is (nearly) a point, drag it whole
    if (nearLo) return "lo";
    if (nearHi) return "hi";
    if (x > xl && x < xh) return "band";
    return "out";
  };
  const cursor = () => {
    if (!lastEv) return;
    const h = dragging ? "out" : hit(lastEv);
    el.classList.toggle("onhandle", h === "lo" || h === "hi");
    el.classList.toggle("onband", h === "band");
  };

  el.setRange = (a, b, st) => { min = a; max = b; step = st ?? step; sLo = lo = quant(lo); sHi = hi = quant(hi); paint(); };
  const setStored = (l: number, h: number) => { l = quant(l); h = quant(h); if (h < l) [l, h] = [h, l]; sLo = lo = l; sHi = hi = h; isNull = false; paint(); };
  Object.defineProperty(el, "lo", { get: () => (isNull ? null : lo), set: (v: number | null) => { if (v === null) isNull = true; else setStored(+v, hi); paint(); } });
  Object.defineProperty(el, "hi", { get: () => (isNull ? null : hi), set: (v: number | null) => { if (v === null) isNull = true; else setStored(lo, +v); paint(); } });
  Object.defineProperty(el, "center", { get: () => (isNull ? null : +((lo + hi) / 2).toFixed(6)) });
  Object.defineProperty(el, "width", { get: () => (isNull ? null : +(hi - lo).toFixed(6)) });
  Object.defineProperty(el, "value", {
    get: () => (isNull ? null : `${lo},${hi}`),
    set: (v: string | [number, number] | null) => {
      if (v === null || v === "null") { isNull = true; paint(); return; }
      const [a, b] = typeof v === "string" ? v.split(",").map(Number) : v;
      setStored(a!, b!);
    },
  });

  /* shift-hover preview: the centre follows the pointer */
  const onGlobal = (e: PointerEvent) => { lastEv = e; if (!e.shiftKey) { endPreview(); return; } if (!dragging) showCentred(atX(e.clientX), sHi - sLo); };
  const startPreview = () => {
    if (el.dataset.nopreview !== undefined || previewing || dragging || settled || isNull || !lastEv) return;
    previewing = true; el.classList.add("hover"); window.addEventListener("pointermove", onGlobal); showCentred(atX(lastEv.clientX), sHi - sLo);
  };
  const endPreview = () => {
    if (!previewing) return;
    previewing = false; el.classList.remove("hover"); window.removeEventListener("pointermove", onGlobal); lo = sLo; hi = sHi; paint(); fire("input");
  };

  /* drags */
  const endDrag = () => {
    if (!dragging) return;
    dragging = false; settled = true; el.classList.remove("locked", "m-lo", "m-hi", "m-band", "m-scale");
    if (downOnThing && !moved) {
      // a click without movement on the band / a handle: unset (nullable) or leave as it was
      if (nullable && mode === "band" && performance.now() - downT <= 500) { lo = sLo; hi = sHi; isNull = true; paint(); fire("input"); fire("change"); }
      else { lo = sLo; hi = sHi; paint(); }
      cursor(); return;
    }
    sLo = lo; sHi = hi; isNull = false; paint(); fire("change"); cursor();
  };
  const cancelDrag = () => {
    if (!dragging) return;
    dragging = false; settled = true; el.classList.remove("locked", "m-lo", "m-hi", "m-band", "m-scale");
    isNull = downWasNull; if (!isNull) { lo = sLo; hi = sHi; } paint(); fire("input"); cursor();
  };
  el.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    endPreview();
    dragging = true; moved = false; downT = performance.now(); downX = e.clientX; downWasNull = isNull;
    downW = sHi - sLo; downC = (sLo + sHi) / 2;
    const v = atX(e.clientX);
    const h = hit(e);
    if (e.altKey) { mode = "scale"; downOnThing = false; if (isNull) downC = quant(v); showScaled(downC, Math.abs(v - downC)); moved = true; }
    else if (h === "lo" || h === "hi") { mode = h; downOnThing = true; downOff = (h === "lo" ? sLo : sHi) - v; }
    else if (h === "band") { mode = "band"; downOnThing = true; downOff = downC - v; }
    else { mode = "band"; downOnThing = false; downOff = 0; if (isNull) { sLo = quant(v - downW / 2); sHi = quant(sLo + downW); } showCentred(v, downW); moved = true; }
    try { el.setPointerCapture(e.pointerId); } catch { /* synthetic events have no active pointer */ } el.classList.add("locked", `m-${mode}`); cursor();
  });
  el.addEventListener("pointermove", (e) => {
    lastEv = e; cursor();
    if (dragging) {
      if (e.buttons === 0) { endDrag(); return; }
      if (downOnThing && !moved && Math.abs(e.clientX - downX) <= 4) return; // dead zone before a handle / band drag starts
      moved = true;
      const v = atX(e.clientX);
      if (mode === "lo") show(Math.min(v + downOff, sHi), sHi);
      else if (mode === "hi") show(sLo, Math.max(v + downOff, sLo));
      else if (mode === "band") showCentred(v + downOff, downW);
      else showScaled(downC, Math.abs(v - downC));
      return;
    }
    if (!e.shiftKey) endPreview();
  });
  el.addEventListener("pointerup", endDrag); el.addEventListener("pointercancel", endDrag);
  el.addEventListener("pointerenter", (e) => { over = true; lastEv = e; cursor(); });
  el.addEventListener("pointerleave", () => { over = false; settled = false; el.classList.remove("onhandle", "onband"); });
  window.addEventListener("keydown", (e) => { if (e.key === "Shift" && over) startPreview(); });
  window.addEventListener("keyup", (e) => { if (e.key === "Shift") endPreview(); });
  window.addEventListener("keydown", (e) => { if (e.key === "Escape") cancelDrag(); });
  window.addEventListener("keydown", (e) => {
    if (e.key === "Backspace" && over && nullable && !dragging && !isNull) { e.preventDefault(); isNull = true; paint(); fire("input"); fire("change"); cursor(); }
  });
  window.addEventListener("blur", () => { endPreview(); endDrag(); });

  /* wheel / arrows: translate, or resize with alt */
  const commit = () => { sLo = lo; sHi = hi; isNull = false; paint(); fire("change"); };
  const nudge = (dir: number) => { const w = sHi - sLo; showCentred((sLo + sHi) / 2 + dir * 0.1 * span(), w); commit(); };
  const grow = (dir: number) => { const c = (sLo + sHi) / 2; showScaled(c, (sHi - sLo) / 2 + dir * 0.05 * span()); commit(); };
  const SCROLL_SIZE = 1000;
  const toScroll = (v: number) => ((v - min) / span()) * SCROLL_SIZE;
  const toSlider = (s: number) => min + (s / SCROLL_SIZE) * span();
  let lastScrollTime = -1e9, currScroll = 0, lastTickEvent = -1e9, scrollAlt = false;
  el.addEventListener("wheel", (e) => {
    if (e.shiftKey || dragging) return;
    e.preventDefault();
    if (isNull) return;
    const now = performance.now();
    const alt = e.altKey;
    if (now - lastScrollTime > 100 || alt !== scrollAlt) currScroll = toScroll(alt ? min + (sHi - sLo) / 2 : (sLo + sHi) / 2);
    lastScrollTime = now; scrollAlt = alt;
    currScroll = Math.min(SCROLL_SIZE, Math.max(0, Math.round(currScroll + (e.deltaY - e.deltaX))));
    if (now - lastTickEvent < 10) return;
    const before = `${lo},${hi}`;
    if (alt) showScaled((sLo + sHi) / 2, toSlider(currScroll) - min); else showCentred(toSlider(currScroll), sHi - sLo);
    if (`${lo},${hi}` !== before) { commit(); lastTickEvent = now; }
  }, { passive: false });
  window.addEventListener("keydown", (e) => {
    if (!over || e.shiftKey || isNull) return;
    const dir = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
    if (!dir) return;
    e.preventDefault();
    if (e.altKey) grow(dir); else nudge(dir);
  });

  paint();
  return el;
}
