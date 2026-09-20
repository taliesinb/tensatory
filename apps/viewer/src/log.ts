// console / error capture for the "L" log modal, the status line, and (dev server) the client log shipped to
// `/__tensatory/log` — the console of clients we cannot inspect (the Safari web app on the Dock): see vite.config.ts.

const $ = (id: string) => document.getElementById(id);

export const LOG: string[] = [];
const push = (level: string, args: unknown[]) => {
  const line = `[${level}] ` + args.map((a) => { try { return typeof a === "object" ? JSON.stringify(a) : String(a); } catch { return String(a); } }).join(" ");
  LOG.push(line);
  ship(line);
};

/*******************************************************/
/* shipping to the dev server: batched every 500 ms (keepalive, so a batch survives a reload), the rest on pagehide */

/** a tag per page load, so interleaved clients can be told apart in the server's file */
export const SESSION = Math.random().toString(36).slice(2, 8);
const SHIP = import.meta.env.DEV;
let outbox: string[] = [], shipTimer: ReturnType<typeof setTimeout> | undefined;
const t0 = performance.now();
function ship(line: string): void {
  if (!SHIP) return;
  outbox.push(`${SESSION} +${((performance.now() - t0) / 1000).toFixed(3)}s ${line}`);
  shipTimer ??= setTimeout(flush, 500);
}
function flush(): void {
  shipTimer = undefined;
  if (!outbox.length) return;
  const body = outbox.join("\n"); outbox = [];
  try { void fetch("/__tensatory/log", { method: "POST", body, keepalive: true, headers: { "content-type": "text/plain" } }).catch(() => {}); } catch { /* ignore */ }
}

export function installLogCapture(): void {
  for (const lvl of ["log", "info", "warn", "error"] as const) {
    const orig = console[lvl].bind(console);
    console[lvl] = (...a: unknown[]) => { push(lvl, a); orig(...a); };
  }
  window.addEventListener("error", (e) => { push("error", [`${e.message} @ ${e.filename}:${e.lineno}:${e.colno}`]); showError(e.message); });
  window.addEventListener("unhandledrejection", (e) => { const m = (e.reason && (e.reason.message || e.reason.stack)) || String(e.reason); push("unhandledrejection", [m]); showError(m); });
  // errors that happened before this module ran (the inline hook in index.html keeps them)
  const early = (window as unknown as { __tensatoryEarly?: { errors: string[]; booted?: boolean } }).__tensatoryEarly;
  if (early) { early.booted = true; for (const m of early.errors) push("early error", [m]); }
  const standalone = matchMedia("(display-mode: standalone)").matches || (navigator as unknown as { standalone?: boolean }).standalone === true;
  push("session", [`${standalone ? "web app (standalone)" : "browser tab"} · ${navigator.userAgent} · ${location.href} · gpu: ${"gpu" in navigator ? "yes" : "no"}`]);
  window.addEventListener("pagehide", flush);
  $("resetBtn")?.addEventListener("click", () => { try { localStorage.clear(); } catch { /* ignore */ } location.replace(location.pathname); });
  $("logBtn")?.addEventListener("click", () => { const t = $("logText"); if (t) t.textContent = LOG.join("\n"); $("logModal")?.classList.add("open"); });
  $("logClose")?.addEventListener("click", () => $("logModal")?.classList.remove("open"));
}

let statusUnpainted = false;
export function status(s: string): void {
  const el = $("status");
  if (el) { el.textContent = s; el.classList.remove("err"); }
  if (s) { push("status", [s]); statusUnpainted = true; }
}
/**
 * A status message was set and has not had a paint yet. rAF callbacks run BEFORE the frame's paint, so a heavy
 * render in the same frame (a remesh, a submit the browser blocks on) would delay the message for seconds: the
 * frame loop yields one frame when this is true. Clears on read.
 */
export function statusAwaitingPaint(): boolean { const p = statusUnpainted; statusUnpainted = false; return p; }

/** an error, shown in red on the status line (the full log is behind the L button) */
export function showError(e: unknown): void {
  const m = e instanceof Error ? e.message : String(e);
  const el = $("status");
  if (el) { el.textContent = `${m} — see L (log)`; el.classList.add("err"); }
  push("error", [e instanceof Error ? e.stack ?? m : m]);
}

/*******************************************************/
/* boot phases: "loading… <what>" with the seconds elapsed once a phase takes more than 2 s, so a stall says where */

let phaseName = "", phaseT0 = 0, phaseTimer: ReturnType<typeof setInterval> | undefined;
/** enter a boot phase (logged with the previous phase's duration); `bootPhase(undefined)` ends the boot */
export function bootPhase(name: string | undefined): void {
  if (phaseName) console.log(`boot: ${phaseName} took ${(performance.now() - phaseT0).toFixed(0)} ms`);
  if (phaseTimer) { clearInterval(phaseTimer); phaseTimer = undefined; }
  phaseName = name ?? ""; phaseT0 = performance.now();
  if (!name) return;
  status(`loading… ${name}`);
  phaseTimer = setInterval(() => {
    const s = Math.round((performance.now() - phaseT0) / 1000);
    if (s >= 2) { const el = $("status"); if (el && !el.classList.contains("err")) el.textContent = `loading… ${phaseName} (${s} s)`; }
    if (s === 5 || s % 15 === 0) console.warn(`boot: still in "${phaseName}" after ${s} s`);
  }, 1000);
}
