// Colormaps (matplotlib polynomial fits, OKLCH hue sweep, turbo), as in the prototype.

export type RGB = [number, number, number];
export type Colormap = (t: number) => RGB;
type Coeffs = RGB[];

const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

const poly = (c: Coeffs): Colormap => (t) => {
  t = clamp01(t);
  const out: RGB = [0, 0, 0];
  for (let ch = 0; ch < 3; ch++) {
    let v = 0;
    for (let i = c.length - 1; i >= 0; i--) v = v * t + c[i]![ch]!;
    out[ch] = clamp01(v);
  }
  return out;
};

const viridis = poly([
  [0.2777273272234177, 0.005407344544966578, 0.3340998053353061],
  [0.1050930431085774, 1.404613529898575, 1.384590162594685],
  [-0.3308618287255563, 0.214847559468213, 0.09509516302823659],
  [-4.634230498983486, -5.799100973351585, -19.33244095627987],
  [6.228269936347081, 14.17993336680509, 56.69055260068105],
  [4.776384997670288, -13.74514537774601, -65.35303263337234],
  [-5.435455855934631, 4.645852612178535, 26.3124486955888],
]);
const plasma = poly([
  [0.05873234392399702, 0.02333670892565664, 0.5433401826748754],
  [2.176514634195958, 0.2383834171260182, 0.7539604599784036],
  [-2.689460476458034, -7.455851135738909, 3.110799939717086],
  [6.130348345893603, 42.3461881477227, -28.51885465332158],
  [-11.10743619062271, -82.66631109428045, 60.13984767418263],
  [10.02306557647065, 71.41361770095349, -54.07218655560067],
  [-3.658713842777788, -22.93153465461149, 18.19190778539828],
]);
const gray: Colormap = (t) => { t = clamp01(t); return [t, t, t]; };
const okhue: Colormap = (t) => {
  t = clamp01(t);
  const L = 0.74, C = 0.13, h = ((20 + 300 * t) * Math.PI) / 180, a = C * Math.cos(h), b = C * Math.sin(h);
  const l_ = L + 0.3963377774 * a + 0.2158037573 * b, m_ = L - 0.1055613458 * a - 0.0638541728 * b, s_ = L - 0.0894841775 * a - 1.291485548 * b;
  const l = l_ ** 3, m = m_ ** 3, s = s_ ** 3;
  const lin = [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s, -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s, -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s];
  return lin.map((v) => { v = clamp01(v); return v <= 0.0031308 ? 12.92 * v : 1.055 * v ** (1 / 2.4) - 0.055; }) as RGB;
};
const turbo: Colormap = (t) => {
  t = clamp01(t);
  const v4 = [1, t, t * t, t ** 3], v2 = [t ** 4, t ** 5];
  const dot = (a: number[], b: number[]) => a.reduce((x, y, i) => x + y * b[i]!, 0);
  return [
    clamp01(dot(v4, [0.13572138, 4.6153926, -42.66032258, 132.13108234]) + dot(v2, [-152.94239396, 59.28637943])),
    clamp01(dot(v4, [0.09140261, 2.19418839, 4.84296658, -14.18503333]) + dot(v2, [4.27729857, 2.82956604])),
    clamp01(dot(v4, [0.1066733, 12.64194608, -60.58204836, 110.36276771]) + dot(v2, [-89.90310912, 27.34824973])),
  ];
};

export const MAPS = ["viridis", "plasma", "gray", "okhue", "turbo"] as const;
export const COLORMAPS: Record<(typeof MAPS)[number], Colormap> = { viridis, plasma, gray, okhue, turbo };
export const cmap = (id: number): Colormap => COLORMAPS[MAPS[((id % MAPS.length) + MAPS.length) % MAPS.length]!];

export const toCss = ([r, g, b]: RGB, a = 1): string =>
  `rgba(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)},${a})`;

export function cssGradient(id: number): string {
  const f = cmap(id), stops: string[] = [];
  for (let i = 0; i <= 24; i++) stops.push(toCss(f(i / 24)));
  return `linear-gradient(90deg, ${stops.join(",")})`;
}

/** a lookup table of n colours for fast per-pixel mapping */
export function lut(f: Colormap, n = 256): Uint8ClampedArray {
  const out = new Uint8ClampedArray(n * 3);
  for (let i = 0; i < n; i++) { const [r, g, b] = f(i / (n - 1)); out[3 * i] = r * 255; out[3 * i + 1] = g * 255; out[3 * i + 2] = b * 255; }
  return out;
}
