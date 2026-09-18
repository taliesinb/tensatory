// console / error capture for the "L" log modal, and the status line.

const $ = (id: string) => document.getElementById(id);

export const LOG: string[] = [];
const push = (level: string, args: unknown[]) =>
  LOG.push(`[${level}] ` + args.map((a) => { try { return typeof a === "object" ? JSON.stringify(a) : String(a); } catch { return String(a); } }).join(" "));

export function installLogCapture(): void {
  for (const lvl of ["log", "info", "warn", "error"] as const) {
    const orig = console[lvl].bind(console);
    console[lvl] = (...a: unknown[]) => { push(lvl, a); orig(...a); };
  }
  window.addEventListener("error", (e) => { push("error", [`${e.message} @ ${e.filename}:${e.lineno}:${e.colno}`]); showError(e.message); });
  window.addEventListener("unhandledrejection", (e) => { const m = (e.reason && (e.reason.message || e.reason.stack)) || String(e.reason); push("unhandledrejection", [m]); showError(m); });
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
