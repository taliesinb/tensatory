// Number / interval formatting shared by the prototype pages.

// formatReal, copied verbatim from packages/core/src/fields/codomain.ts (importing it pulls all of core + zod into the static build)
const SUP = "⁰¹²³⁴⁵⁶⁷⁸⁹";
const superscript = (n: number): string => `${n < 0 ? "⁻" : ""}${String(Math.abs(n)).split("").map((d) => SUP[+d]!).join("")}`;
export function formatReal(v: number, digits = 3): string {
  if (!Number.isFinite(v)) return v > 0 ? "∞" : v < 0 ? "−∞" : "NaN";
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e4 || a < 1e-3) {
    const e = Math.floor(Math.log10(a));
    let m = v / Math.pow(10, e);
    let ms = m.toPrecision(digits);
    if (Math.abs(+ms) >= 10) { m /= 10; ms = m.toPrecision(digits); return `${+ms}·10${superscript(e + 1)}`; }
    return `${+ms}·10${superscript(e)}`;
  }
  return String(+v.toPrecision(digits));
}

/** Interval readout (fixed-width column, never units): "x" for a point, "±y" when symmetric about 0,
 *  "≥ x" / "≤ y" for half intervals, else "x to y". Two numbers get 3 significant digits, one number gets 4. */
export function formatInterval(lo: number | null, hi: number | null): string {
  if (lo === null && hi === null) return "";
  if (lo === null) return `≤ ${formatReal(hi!, 4)}`;
  if (hi === null) return `≥ ${formatReal(lo, 4)}`;
  if (lo === hi) return formatReal(lo, 4);
  if (lo === -hi) return `±${formatReal(hi, 4)}`;
  return `${formatReal(lo, 3)} to ${formatReal(hi, 3)}`;
}
