// Framework-free widgets, ported from the loss-landscape prototype.
// Controls are plain <div>s configured by data- attributes; sliders expose a
// `.value` (string | null) property and dispatch 'input' (live) / 'change'
// (committed) events, like native inputs.

export interface ValueControl extends HTMLElement {
  value: string | null;
}
export interface SliderEl extends ValueControl {
  setRange(min: number, max: number, step?: number): void;
}

const $ = (id: string) => document.getElementById(id)!;

/*******************************************************/
/* tooltips (0.25 s delay) on any [data-tip] */

export function installTooltips(root: ParentNode = document): void {
  const tip = $("tip");
  let timer: ReturnType<typeof setTimeout> | undefined;
  for (const el of root.querySelectorAll<HTMLElement>("[data-tip]")) {
    el.addEventListener("pointerenter", () => {
      if (!el.dataset.tip) return; // data-tip="" reserves a tooltip whose text is set later
      timer = setTimeout(() => {
        tip.textContent = el.dataset.tip ?? "";
        tip.style.display = "block";
        const r = el.getBoundingClientRect(), tw = tip.offsetWidth, th = tip.offsetHeight;
        let x = r.left, y = r.bottom + 6;
        if (x + tw > window.innerWidth - 8) x = window.innerWidth - 8 - tw;
        if (y + th > window.innerHeight - 8) y = r.top - th - 6;
        tip.style.left = `${x}px`; tip.style.top = `${y}px`;
      }, 250);
    });
    el.addEventListener("pointerleave", () => { clearTimeout(timer); tip.style.display = "none"; });
  }
}

/*******************************************************/
/* collapsible panels: click the strip title; remembered globally. A panel that is `.collapsed` in
   the HTML starts closed until the user has toggled it once. */

export function installCollapsiblePanels(storageKey: string, onToggle?: () => void): void {
  let saved: Record<string, boolean> = {};
  try { saved = JSON.parse(localStorage.getItem(storageKey) ?? "{}"); } catch { /* ignore */ }
  for (const panel of document.querySelectorAll<HTMLElement>(".panel")) {
    const title = panel.querySelector<HTMLElement>(".strip .title");
    if (!title) continue;
    const key = title.textContent!.trim();
    if (key in saved) panel.classList.toggle("collapsed", saved[key]);
    title.addEventListener("click", () => {
      panel.classList.toggle("collapsed");
      saved[key] = panel.classList.contains("collapsed");
      localStorage.setItem(storageKey, JSON.stringify(saved));
      onToggle?.();
    });
  }
}

/*******************************************************/
/* ticks: a glyph label wrapping a hidden checkbox */

const tickSyncs: (() => void)[] = [];
export function installTicks(): void {
  for (const lab of document.querySelectorAll<HTMLLabelElement>(".tick")) {
    const cb = lab.querySelector("input")!;
    const sync = () => lab.classList.toggle("on", cb.checked);
    cb.addEventListener("change", sync);
    sync();
    tickSyncs.push(sync);
  }
}
export const syncTicks = (): void => tickSyncs.forEach((f) => f());

/*******************************************************/
/* compact continuous slider */

export const fmtNum = (v: number): string =>
  Math.abs(v) >= 1000 ? `${(v / 1000).toString().replace(/\.0$/, "")}k` : String(v);

// A grey bar with a 3px blue tick at the value. Drag to scrub (pointer captured; clips past the ends);
// shift-press while hovering previews the value until shift is released. data-nullable: a click (no drag,
// <= 500 ms) on the current value unsets it (.value === null); clicking an unset bar sets it where you
// clicked. Wheel scrolls a clamped "document" (1000 positions across [min,max]); arrows nudge 10%.
export function makeSlider(el0: HTMLElement): SliderEl {
  const el = el0 as SliderEl;
  let min = +el.dataset.min!, max = +el.dataset.max!, step = +(el.dataset.step ?? 0);
  const nullable = el.dataset.nullable !== undefined;
  let stored = +(el.dataset.value ?? min), shown = stored, isNull = el.dataset.value === "null", dragging = false, moved = false;
  let downV = 0, downX = 0, downT = 0, downOnHandle = false, downWasNull = false;
  let settled = false, over = false, previewing = false, lastEv: PointerEvent | null = null;
  const was = document.createElement("div"); was.className = "was"; el.appendChild(was);
  const fill = document.createElement("div"); fill.className = "fill"; el.appendChild(fill);
  const quant = (v: number) => { v = Math.min(max, Math.max(min, v)); if (step) v = min + Math.round((v - min) / step) * step; return +v.toFixed(6); };
  const paint = () => {
    fill.style.left = `${((shown - min) / (max - min)) * 100}%`;
    was.style.left = `${((stored - min) / (max - min)) * 100}%`;
    was.style.display = shown !== stored && !isNull ? "block" : "none";
    el.classList.toggle("null", isNull);
  };
  const fire = (t: string) => el.dispatchEvent(new Event(t));
  const atMouse = (e: PointerEvent) => { const r = el.getBoundingClientRect(); return quant(min + ((e.clientX - r.left) / r.width) * (max - min)); };
  const show = (v: number) => { if (v !== shown || isNull) { shown = v; isNull = false; paint(); fire("input"); } };
  const onHandle = (e: PointerEvent) => { if (isNull) return false; const r = el.getBoundingClientRect(); return Math.abs(e.clientX - (r.left + ((stored - min) / (max - min)) * r.width)) <= 4; };
  const cursor = () => { if (lastEv) el.classList.toggle("onhandle", !dragging && onHandle(lastEv)); };
  el.setRange = (a, b, st) => { min = a; max = b; step = st ?? step; stored = shown = quant(shown); paint(); };
  Object.defineProperty(el, "value", {
    get: () => (isNull ? null : String(shown)),
    set: (v: string | number | null) => { if (v === null || v === "null") isNull = true; else { stored = shown = quant(+v); isNull = false; } paint(); },
  });
  const onGlobal = (e: PointerEvent) => { lastEv = e; if (!e.shiftKey) { endPreview(); return; } if (!dragging) show(atMouse(e)); };
  const startPreview = () => { if (el.dataset.nopreview !== undefined || previewing || dragging || settled || !lastEv) return; previewing = true; el.classList.add("hover"); window.addEventListener("pointermove", onGlobal); show(atMouse(lastEv)); };
  const endPreview = () => { if (!previewing) return; previewing = false; el.classList.remove("hover"); window.removeEventListener("pointermove", onGlobal); shown = stored; paint(); fire("input"); };
  const endDrag = () => {
    if (!dragging) return;
    dragging = false; settled = true; el.classList.remove("locked");
    if (downOnHandle && !moved) {
      if (nullable && performance.now() - downT <= 500) { shown = stored; isNull = true; paint(); fire("input"); fire("change"); }
      else { shown = stored; paint(); }
      cursor(); return;
    }
    stored = shown; isNull = false; paint(); fire("change"); cursor();
  };
  const cancelDrag = () => { if (!dragging) return; dragging = false; settled = true; el.classList.remove("locked"); isNull = downWasNull; if (!isNull) shown = stored; paint(); fire("input"); cursor(); };
  el.addEventListener("pointerdown", (e) => { endPreview(); downOnHandle = onHandle(e); dragging = true; moved = false; downT = performance.now(); downX = e.clientX; downV = atMouse(e); downWasNull = isNull; el.setPointerCapture(e.pointerId); el.classList.add("locked"); if (isNull) stored = downV; if (!downOnHandle) show(downV); });
  el.addEventListener("pointermove", (e) => {
    lastEv = e; el.classList.toggle("onhandle", !dragging && onHandle(e));
    if (dragging) {
      if (e.buttons === 0) { endDrag(); return; }
      if (downOnHandle) { if (Math.abs(e.clientX - downX) > 4) moved = true; if (moved) show(atMouse(e)); return; }
      const v = atMouse(e); if (v !== downV) moved = true; show(v); return;
    }
    if (!e.shiftKey) endPreview();
  });
  el.addEventListener("pointerup", endDrag); el.addEventListener("pointercancel", endDrag);
  el.addEventListener("pointerenter", (e) => { over = true; lastEv = e; });
  el.addEventListener("pointerleave", () => { over = false; settled = false; el.classList.remove("onhandle"); });
  window.addEventListener("keydown", (e) => { if (e.key === "Shift" && over) startPreview(); });
  window.addEventListener("keyup", (e) => { if (e.key === "Shift") endPreview(); });
  window.addEventListener("keydown", (e) => { if (e.key === "Escape") cancelDrag(); });
  window.addEventListener("keydown", (e) => { if (e.key === "Backspace" && over && nullable && !dragging && !isNull) { e.preventDefault(); isNull = true; paint(); fire("input"); fire("change"); cursor(); } });
  window.addEventListener("blur", () => { endPreview(); endDrag(); });
  const nudge = (dir: number) => { stored = shown = quant((isNull ? stored : shown) + dir * 0.1 * (max - min)); isNull = false; paint(); fire("input"); fire("change"); };
  const SCROLL_SIZE = 1000;
  const toScroll = (v: number) => ((v - min) / (max - min)) * SCROLL_SIZE;
  const toSlider = (s: number) => quant(min + (s / SCROLL_SIZE) * (max - min));
  let lastScrollTime = -1e9, currScrollValue = 0, lastTickEvent = -1e9;
  el.addEventListener("wheel", (e) => {
    if (e.shiftKey || dragging) return;
    e.preventDefault();
    const now = performance.now();
    if (now - lastScrollTime > 100) currScrollValue = toScroll(isNull ? min : shown);
    lastScrollTime = now;
    if (isNull) return;
    currScrollValue = Math.min(SCROLL_SIZE, Math.max(0, Math.round(currScrollValue + (e.deltaY - e.deltaX))));
    const nv = toSlider(currScrollValue);
    if (nv !== shown && now - lastTickEvent >= 10) { stored = shown = nv; isNull = false; paint(); fire("input"); fire("change"); lastTickEvent = now; }
  }, { passive: false });
  window.addEventListener("keydown", (e) => { if (!over || e.shiftKey) return; if (e.key === "ArrowRight") { e.preventDefault(); nudge(1); } else if (e.key === "ArrowLeft") { e.preventDefault(); nudge(-1); } });
  paint();
  return el;
}

/*******************************************************/
/* wheel -> one step per gesture */

export function wheelStepper(onStep: (dir: 1 | -1) => void): (e: WheelEvent) => void {
  const TICK_MS = 250, BURST_MS = 100;
  let lastMag = 0, lastT = -1e9, lastTick = -1e9;
  return (e) => {
    e.preventDefault();
    const pick = Math.abs(e.deltaX) >= Math.abs(e.deltaY) ? -e.deltaX : e.deltaY;
    const mag = Math.abs(pick); if (mag <= 1) return;
    const now = performance.now();
    const recent = now - lastT < BURST_MS;
    const magOK = recent ? mag > lastMag : mag >= 2;
    if (magOK && now - lastTick >= TICK_MS) { onStep(pick >= 0 ? 1 : -1); lastTick = now; }
    lastMag = mag; lastT = now;
  };
}

/*******************************************************/
/* discrete slider: data-values="0,2,4" data-value [data-nullable] */

export function makeDiscreteSlider(el0: HTMLElement): ValueControl {
  const el = el0 as ValueControl;
  const values = el.dataset.values!.split(",").map(Number), nullable = el.dataset.nullable !== undefined;
  let value: number | null = el.dataset.value !== undefined && el.dataset.value !== "" ? +el.dataset.value : null;
  const nearest = (v: number) => values.reduce((a, b) => (Math.abs(b - v) < Math.abs(a - v) ? b : a));
  const segs = values.map((v) => { const s = document.createElement("div"); s.className = "seg"; s.textContent = fmtNum(v); s.style.flexGrow = String(fmtNum(v).length + 1); el.appendChild(s); return s; });
  const paint = () => segs.forEach((s, i) => s.classList.toggle("on", values[i] === value));
  const fire = (t: string) => el.dispatchEvent(new Event(t));
  const set = (v: number | null) => { value = v; paint(); fire("input"); fire("change"); };
  segs.forEach((s, i) => s.addEventListener("click", () => set(nullable && values[i] === value ? null : values[i]!)));
  let over = false;
  el.addEventListener("pointerenter", () => (over = true)); el.addEventListener("pointerleave", () => (over = false));
  const step = (d: number) => { const i = value === null ? -1 : values.indexOf(value); const j = i < 0 ? (d > 0 ? 0 : values.length - 1) : Math.max(0, Math.min(values.length - 1, i + d)); set(values[j]!); };
  const wheel = wheelStepper(step);
  el.addEventListener("wheel", (e) => { if (e.shiftKey) return; wheel(e); }, { passive: false });
  window.addEventListener("keydown", (e) => { if (!over || e.shiftKey) return; if (e.key === "ArrowRight") { e.preventDefault(); step(1); } else if (e.key === "ArrowLeft") { e.preventDefault(); step(-1); } else if (e.key === "Backspace" && nullable && value !== null) { e.preventDefault(); set(null); } });
  Object.defineProperty(el, "value", {
    get: () => (value === null ? null : String(value)),
    set: (v: string | number | null) => { value = v === null || v === "null" ? null : nearest(+v); paint(); },
  });
  paint();
  return el;
}

/*******************************************************/
/* choice: data-options="a,b,c" data-value="a" — a discrete slider over named options (never null); the same .seg look */

export function makeChoice(el0: HTMLElement): ValueControl {
  const el = el0 as ValueControl;
  const options = el.dataset.options!.split(",");
  const tips = el.dataset.tips?.split("|") ?? [];
  let value = options.includes(el.dataset.value ?? "") ? el.dataset.value! : options[0]!;
  // segment widths follow the labels, as the discrete sliders do
  const segs = options.map((o, i) => { const s = document.createElement("div"); s.className = "seg"; s.textContent = o; s.style.flexGrow = String(o.length + 1); if (tips[i]) s.dataset.tip = tips[i]!; el.appendChild(s); return s; });
  const paint = () => segs.forEach((s, i) => s.classList.toggle("on", options[i] === value));
  const fire = (t: string) => el.dispatchEvent(new Event(t));
  const set = (v: string) => { if (v === value) return; value = v; paint(); fire("input"); fire("change"); };
  segs.forEach((s, i) => s.addEventListener("click", () => set(options[i]!)));
  let over = false;
  el.addEventListener("pointerenter", () => (over = true)); el.addEventListener("pointerleave", () => (over = false));
  const step = (d: number) => set(options[Math.max(0, Math.min(options.length - 1, options.indexOf(value) + d))]!);
  const wheel = wheelStepper(step);
  el.addEventListener("wheel", (e) => { if (e.shiftKey) return; wheel(e); }, { passive: false });
  window.addEventListener("keydown", (e) => { if (!over || e.shiftKey) return; if (e.key === "ArrowRight") { e.preventDefault(); step(1); } else if (e.key === "ArrowLeft") { e.preventDefault(); step(-1); } });
  Object.defineProperty(el, "value", {
    get: () => value,
    set: (v: string | null) => { if (v !== null && options.includes(v)) { value = v; paint(); } },
  });
  paint();
  return el;
}

/*******************************************************/
/* tab bar: a row of .tab buttons, one active */

export interface TabItem { value: string; label: string; tip?: string; disabled?: boolean }

export function tabBar(el: HTMLElement, items: TabItem[], current: string | null, onPick: (value: string) => void): void {
  el.replaceChildren();
  for (const it of items) {
    const b = document.createElement("button");
    b.textContent = it.label; b.title = it.tip ?? "";
    b.className = `tab${it.value === current ? " active" : ""}`;
    b.disabled = Boolean(it.disabled);
    b.onclick = () => onPick(it.value);
    el.appendChild(b);
  }
}
