// Interval slider — a two-ended sibling of the compact slider in widgets.ts (prototyped in apps/ui-proto).
//
//   <div class="isl" data-min="0" data-max="1" data-step="0.001" data-lo="0.2" data-hi="0.6"></div>
//
// The root element owns its state and creates its own presentation children, exactly like `.sl` creates
// `.fill` / `.was`:  .band  the filled part, .lo / .hi  3px edge handles, .was  dotted ghosts of the committed
// ends while a drag or preview is in flight. The centre is derived, never stored or drawn.
//
// Either end may be null (data-lo / data-hi = "null"; absent = the range's end), giving four kinds:
//   full  (lo, hi)      band [lo, hi], two handles
//   half  (lo, null)    band [lo, max] fading out to the open end, one handle   (likewise (null, hi))
//   none  (null, null)  empty bar
//
// Gestures. A press is a CLICK if released within CLICK_MS (150 ms); until then nothing moves and the cursor keeps
// its hover shape. Held longer it is a DRAG: the cursor changes and, from then on, pointer motion counts (movement
// during the dead zone is ignored; jumps — press outside, alt-press — are applied when the drag arms or on the click).
//   full:  press a handle -> drag that end (clamped so lo <= hi); click it -> destroy that end;
//          dragging it off the bar's end also destroys it (-> half), live, until the pointer comes back
//          press the band -> translate (width kept, clipped at the ends); click it -> none
//          press outside -> centre jumps there (a click does the same)
//          alt-press -> symmetric resize about the centre (also the way to grow a point interval, whose
//          handles overlap: a plain press there drags it whole)
//          shift-hover -> preview: centre follows the pointer;  wheel translates, alt-wheel resizes;
//          <- -> nudge the centre 10 %, alt+arrows narrow / widen
//   half:  press ANYWHERE and drag -> the existing end moves with the pointer (relative); off either side -> none
//          click the filled part -> the other end appears there;  click the handle -> destroy it (-> none)
//          shift-hover -> preview: the existing end follows the pointer;  wheel / arrows move it
//   none:  press and drag -> a new full interval spanning the dragged range, live; release commits;
//          releasing past either end of the bar leaves that side open -> a half interval
//          click -> a small full interval (CLICK_W = 10 % of the range) centred at the pointer
//   any:   Escape cancels the drag (restores the committed ends);  Backspace -> none
// Colormap variant (class "cmap", drawn as a bracket |‾‾‾| : a top line + the two edges): handles move ONLY by
// dragging; a press anywhere else drags the interval / the existing end relatively (no jump). Single clicks:
//   a handle -> destroy it;  the top line of a half -> the missing end appears there;
//   inside the box -> 'modetoggle' event with detail "span";  outside -> 'modetoggle' with detail "low" / "high"
//   (no event and no pointer cursor with data-notoggle);  an empty bar -> a small interval, as above.
//   Dragging the top line translates the whole bracket, like dragging inside it. The consumer (cmapInterval.ts)
//   owns what those toggles mean.
// Events: 'input' while the shown interval moves (drag / preview), 'change' when it is committed.

/** a press released within this many ms is a click, not a drag; until then motion is ignored and the cursor is unchanged */
const CLICK_MS = 150;
/** width of the interval a click on an empty bar creates, as a fraction of the range */
const CLICK_W = 0.1;

export interface IntervalEl extends HTMLElement {
  lo: number | null;
  hi: number | null;
  /** "lo,hi" with "null" for a missing end (e.g. "0.2,null"); null when both are missing */
  value: string | null;
  /** centre / width of a full interval, else null */
  readonly center: number | null;
  readonly width: number | null;
  readonly min: number;
  readonly max: number;
  setRange(min: number, max: number, step?: number): void;
}

type End = "lo" | "hi";
type Mode = End | "band" | "scale" | "create";
type Region = "lo" | "hi" | "band" | "frame" | "out"; // what was under the pointer at press time ("frame": a cmap half's top line)

export function makeIntervalSlider(el0: HTMLElement): IntervalEl {
  const el = el0 as IntervalEl;
  let min = +el.dataset.min!, max = +el.dataset.max!, step = +(el.dataset.step ?? 0);
  const cmap = el.classList.contains("cmap"), notoggle = el.dataset.notoggle !== undefined;
  const span = () => max - min;
  const quant = (v: number) => { v = Math.min(max, Math.max(min, v)); if (step) v = min + Math.round((v - min) / step) * step; return +v.toFixed(6); };
  const num = (s: string | undefined, dflt: number): number | null => { if (s === "null") return null; const v = s === undefined ? dflt : +s; return Number.isFinite(v) ? quant(v) : dflt; };

  // stored = committed, shown = what is painted (differs during drags / previews)
  let sLo = num(el.dataset.lo, min), sHi = num(el.dataset.hi, max);
  if (sLo !== null && sHi !== null && sHi < sLo) [sLo, sHi] = [sHi, sLo];
  let lo = sLo, hi = sHi;
  // dragging: the button is down.  armed: the press has outlived the click dead zone (classes on, motion counts).
  // moved: the shown interval was changed by the drag.  jump: a change to apply when the drag arms, or on a click.
  let dragging = false, armed = false, moved = false, mode: Mode = "band", region: Region = "out";
  let downT = 0, downV = 0, downOff = 0, downW = 0, downC = 0, armTimer = 0, jump: (() => void) | null = null;
  let settled = false, over = false, previewing = false, lastEv: PointerEvent | null = null;

  const mk = (cls: string) => { const d = document.createElement("div"); d.className = cls; el.appendChild(d); return d; };
  const wasLo = mk("was"), wasHi = mk("was"), band = mk("band"), hLo = mk("lo"), hHi = mk("hi");

  const kindOf = (l: number | null, h: number | null) => (l !== null && h !== null ? "full" : l !== null ? "lo" : h !== null ? "hi" : "none");
  const pct = (v: number) => `${((v - min) / span()) * 100}%`;
  const paint = () => {
    const k = kindOf(lo, hi);
    el.classList.toggle("full", k === "full"); el.classList.toggle("half-lo", k === "lo"); el.classList.toggle("half-hi", k === "hi"); el.classList.toggle("none", k === "none");
    const l = lo ?? min, h = hi ?? max;
    band.style.left = pct(l); band.style.width = pct(min + h - l);
    hLo.style.left = pct(l); hHi.style.left = pct(h);
    const ghost = (w: HTMLElement, stored: number | null, shown: number | null) => {
      const on = stored !== null && stored !== shown;
      w.style.display = on ? "block" : "none"; if (on) w.style.left = pct(stored!);
    };
    ghost(wasLo, sLo, lo); ghost(wasHi, sHi, hi);
  };
  const fire = (t: string) => el.dispatchEvent(new Event(t));
  const rect = () => el.getBoundingClientRect();
  const atX = (clientX: number) => { const r = rect(); return min + ((clientX - r.left) / r.width) * span(); }; // unquantized
  const xOf = (v: number) => { const r = rect(); return r.left + ((v - min) / span()) * r.width; };

  /** show (l, h) — quantized, ordered — and fire input if anything changed */
  const show = (l: number | null, h: number | null) => {
    if (l !== null) l = quant(l); if (h !== null) h = quant(h);
    if (l !== null && h !== null && h < l) [l, h] = [h, l];
    if (l !== lo || h !== hi) { lo = l; hi = h; paint(); fire("input"); }
  };
  /** full: translate so the centre is at c with width w, clipped at the ends */
  const showCentred = (c: number, w: number) => { let l = c - w / 2; l = Math.max(min, Math.min(max - w, l)); show(l, l + w); };
  /** full: symmetric resize about c to half-width hw */
  const showScaled = (c: number, hw: number) => { hw = Math.max(0, Math.min(hw, c - min, max - c)); show(c - hw, c + hw); };
  /** move one end to v (the other end, if any, bounds it) */
  const showEnd = (e: End, v: number) => { if (e === "lo") show(hi === null ? v : Math.min(v, hi), hi); else show(lo, lo === null ? v : Math.max(v, lo)); };
  const commit = () => { sLo = lo; sHi = hi; paint(); fire("change"); };
  const revert = () => { lo = sLo; hi = sHi; paint(); };

  /** what is under the pointer, judged against the committed state */
  const hit = (e: PointerEvent): Region => {
    const k = kindOf(sLo, sHi), x = e.clientX;
    if (k === "none") return "out";
    const nearLo = sLo !== null && Math.abs(x - xOf(sLo)) <= 4, nearHi = sHi !== null && Math.abs(x - xOf(sHi)) <= 4;
    if (nearLo && nearHi) return "band"; // point interval: handles overlap, drag it whole
    if (nearLo) return "lo";
    if (nearHi) return "hi";
    if (!(x > xOf(sLo ?? min) && x < xOf(sHi ?? max))) return "out";
    if (cmap && k !== "full" && e.clientY < rect().top + 3) return "frame"; // the top line (2px above the bar, plus a little inside it)
    return "band";
  };
  const cursor = () => {
    if (!lastEv) return;
    const h = armed ? null : hit(lastEv), k = kindOf(sLo, sHi); // an unarmed press keeps the hover cursor
    el.classList.toggle("onhandle", h === "lo" || h === "hi");
    el.classList.toggle("onband", h === "band" && k === "full" && !cmap);
    el.classList.toggle("onfill", h === "frame" || (h === "band" && k !== "full" && !cmap));
    el.classList.toggle("ontoggle", cmap && !notoggle && (h === "band" || (h === "out" && k !== "none")));
  };

  /* public API */
  el.setRange = (a, b, st) => { min = a; max = b; step = st ?? step; sLo = lo = lo === null ? null : quant(lo); sHi = hi = hi === null ? null : quant(hi); paint(); };
  const setStored = (l: number | null, h: number | null) => { if (l !== null) l = quant(l); if (h !== null) h = quant(h); if (l !== null && h !== null && h < l) [l, h] = [h, l]; sLo = lo = l; sHi = hi = h; paint(); };
  Object.defineProperty(el, "lo", { get: () => lo, set: (v: number | null) => setStored(v, hi) });
  Object.defineProperty(el, "hi", { get: () => hi, set: (v: number | null) => setStored(lo, v) });
  Object.defineProperty(el, "min", { get: () => min });
  Object.defineProperty(el, "max", { get: () => max });
  Object.defineProperty(el, "center", { get: () => (lo !== null && hi !== null ? +((lo + hi) / 2).toFixed(6) : null) });
  Object.defineProperty(el, "width", { get: () => (lo !== null && hi !== null ? +(hi - lo).toFixed(6) : null) });
  const parseEnd = (s: string | number | null | undefined): number | null => (s === null || s === undefined || s === "null" || s === "" ? null : +s);
  Object.defineProperty(el, "value", {
    get: () => (lo === null && hi === null ? null : `${lo ?? "null"},${hi ?? "null"}`),
    set: (v: string | [number | null, number | null] | null) => {
      if (v === null || v === "null") { setStored(null, null); return; }
      const [a, b] = typeof v === "string" ? v.split(",") : v;
      setStored(parseEnd(a), parseEnd(b));
    },
  });

  /* shift-hover preview: full -> the centre follows the pointer; half -> the existing end does */
  const previewAt = (clientX: number) => {
    const k = kindOf(sLo, sHi), v = atX(clientX);
    if (k === "full") showCentred(v, sHi! - sLo!); else if (k === "lo") show(v, null); else if (k === "hi") show(null, v);
  };
  const onGlobal = (e: PointerEvent) => { lastEv = e; if (!e.shiftKey) { endPreview(); return; } if (!dragging) previewAt(e.clientX); };
  const startPreview = () => {
    if (el.dataset.nopreview !== undefined || previewing || dragging || settled || !lastEv || kindOf(sLo, sHi) === "none") return;
    previewing = true; el.classList.add("hover"); window.addEventListener("pointermove", onGlobal); previewAt(lastEv.clientX);
  };
  const endPreview = () => {
    if (!previewing) return;
    previewing = false; el.classList.remove("hover"); window.removeEventListener("pointermove", onGlobal); revert(); fire("input");
  };

  /* drags */
  const MODES = ["m-lo", "m-hi", "m-band", "m-scale", "m-create"];
  /** the press has outlived the click dead zone: it is a drag — show it (cursor, colours) and apply a pending jump */
  const arm = () => {
    if (!dragging || armed) return;
    armed = true; clearTimeout(armTimer); el.classList.add("locked", `m-${mode}`);
    if (jump) { jump(); jump = null; moved = true; }
    cursor();
  };
  /** the press was released before arming: a click */
  const click = () => {
    const k = kindOf(sLo, sHi);
    const addEnd = () => { if (k === "lo") { lo = sLo; hi = quant(downV); } else { lo = quant(downV); hi = sHi; } commit(); };
    if (k === "none") showCentred(downV, CLICK_W * span());                  // empty bar: a small interval appears at the pointer
    else if (region === "lo") { lo = null; hi = sHi; }                         // a handle: destroy that end
    else if (region === "hi") { lo = sLo; hi = null; }
    else if (region === "frame") { addEnd(); return; }                        // cmap half: the frame -> the other end appears there
    else if (region === "band" && k !== "full" && !cmap) { addEnd(); return; } // plain half: the filled part -> likewise
    else if (cmap) {                                                           // cmap: clicks toggle the consumer's modes
      revert();
      const what = region === "band" ? "span" : downV < (sLo ?? min) ? "low" : "high";
      if (!notoggle) el.dispatchEvent(new CustomEvent("modetoggle", { detail: what }));
      return;
    }
    else if (region === "band") lo = hi = null;                                // plain full: the band -> none
    else if (jump) { jump(); jump = null; }                                    // plain full, outside / alt: the jump happens anyway
    else { revert(); return; }
    commit();
  };
  const endDrag = (asClick = true) => {
    if (!dragging) return;
    const wasArmed = armed;
    dragging = false; armed = false; settled = true; clearTimeout(armTimer); el.classList.remove("locked", ...MODES);
    if (!wasArmed) { if (asClick) click(); else revert(); }
    else if (moved) commit();
    else revert();
    jump = null; cursor();
  };
  const cancelDrag = () => { if (!dragging) return; dragging = false; armed = false; settled = true; clearTimeout(armTimer); jump = null; el.classList.remove("locked", ...MODES); revert(); fire("input"); cursor(); };
  el.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    endPreview();
    dragging = true; armed = false; moved = false; jump = null; downT = performance.now(); downV = atX(e.clientX); region = hit(e);
    const k = kindOf(sLo, sHi);
    if (k === "none") { mode = "create"; }
    else if (k === "lo" || k === "hi") { mode = k; downOff = (k === "lo" ? sLo! : sHi!) - downV; } // press anywhere: move the existing end, relatively
    else if (e.altKey) { mode = "scale"; downC = (sLo! + sHi!) / 2; jump = () => showScaled(downC, Math.abs(downV - downC)); }
    else if (region === "lo" || region === "hi") { mode = region; downOff = (region === "lo" ? sLo! : sHi!) - downV; }
    else if (region === "band" || region === "frame" || cmap) { mode = "band"; downW = sHi! - sLo!; downOff = (sLo! + sHi!) / 2 - downV; } // drag the whole interval (band or frame), relatively
    else { mode = "band"; downW = sHi! - sLo!; downOff = 0; jump = () => showCentred(downV, downW); } // plain, press outside: the centre jumps there
    try { el.setPointerCapture(e.pointerId); } catch { /* synthetic events have no active pointer */ }
    armTimer = window.setTimeout(arm, CLICK_MS); cursor();
  });
  el.addEventListener("pointermove", (e) => {
    lastEv = e; cursor();
    if (dragging) {
      if (e.buttons === 0) { endDrag(); return; }
      if (!armed) { if (performance.now() - downT < CLICK_MS) return; arm(); } // dead zone: motion within a click is ignored
      moved = true;
      const v = atX(e.clientX), r = rect(), offL = e.clientX < r.left, offR = e.clientX > r.right;
      if (mode === "lo" || mode === "hi") {
        // dragging an end off the bar destroys it: a full interval loses that end (-> half), a half loses its only end (-> none)
        const k = kindOf(sLo, sHi);
        if (k !== "full" && (offL || offR)) show(null, null);
        else if (mode === "lo" && offL) show(null, sHi);
        else if (mode === "hi" && offR) show(sLo, null);
        else showEnd(mode, v + downOff);
      }
      else if (mode === "band") showCentred(v + downOff, downW);
      else if (mode === "scale") showScaled(downC, Math.abs(v - downC));
      else { // create: the dragged range; past the bar's end the far side stays open -> a half interval
        if (offR) show(downV, null); else if (offL) show(null, downV); else show(Math.min(downV, v), Math.max(downV, v));
      }
      return;
    }
    if (!e.shiftKey) endPreview();
  });
  el.addEventListener("pointerup", () => endDrag()); el.addEventListener("pointercancel", () => endDrag(false));
  el.addEventListener("pointerenter", (e) => { over = true; lastEv = e; cursor(); });
  el.addEventListener("pointerleave", () => { over = false; settled = false; el.classList.remove("onhandle", "onband", "onfill", "ontoggle"); });
  window.addEventListener("keydown", (e) => { if (e.key === "Shift" && over) startPreview(); });
  window.addEventListener("keyup", (e) => { if (e.key === "Shift") endPreview(); });
  window.addEventListener("keydown", (e) => { if (e.key === "Escape") cancelDrag(); });
  window.addEventListener("keydown", (e) => {
    if (e.key === "Backspace" && over && !dragging && kindOf(sLo, sHi) !== "none") { e.preventDefault(); lo = hi = null; paint(); fire("input"); commit(); cursor(); }
  });
  window.addEventListener("blur", () => { endPreview(); endDrag(false); }); // losing focus is not a click

  /* wheel / arrows: full -> translate (alt: resize); half -> move the existing end */
  const stepBy = (dv: number, alt: boolean) => {
    const k = kindOf(sLo, sHi);
    if (k === "full") { if (alt) showScaled((sLo! + sHi!) / 2, (sHi! - sLo!) / 2 + dv / 2); else showCentred((sLo! + sHi!) / 2 + dv, sHi! - sLo!); }
    else if (k === "lo") show(sLo! + dv, null);
    else if (k === "hi") show(null, sHi! + dv);
    if (lo !== sLo || hi !== sHi) commit();
  };
  const SCROLL_SIZE = 1000;
  let lastScrollTime = -1e9, currScroll = 0, lastTickEvent = -1e9, scrollAlt = false;
  el.addEventListener("wheel", (e) => {
    if (e.shiftKey || dragging) return;
    e.preventDefault();
    if (kindOf(sLo, sHi) === "none") return;
    const now = performance.now(), alt = e.altKey;
    if (now - lastScrollTime > 100 || alt !== scrollAlt) currScroll = 0;
    lastScrollTime = now; scrollAlt = alt;
    currScroll += e.deltaY - e.deltaX;
    if (now - lastTickEvent < 10) return;
    const ticks = Math.trunc(currScroll); if (!ticks) return;
    currScroll -= ticks;
    stepBy((ticks / SCROLL_SIZE) * span(), alt); lastTickEvent = now;
  }, { passive: false });
  window.addEventListener("keydown", (e) => {
    if (!over || e.shiftKey) return;
    const dir = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
    if (!dir) return;
    e.preventDefault();
    stepBy(dir * 0.1 * span(), e.altKey);
  });

  paint();
  return el;
}
