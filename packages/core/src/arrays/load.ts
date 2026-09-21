// Loading stored arrays: bytes come from the environment (a ByteSource: fetch
// in the viewer, readFile in tests), formats are decoded here. Isomorphic — no
// fetch, no fs; only web-standard APIs (DecompressionStream, Response, Blob),
// which Node ≥ 18 also provides.
//
// The FORMAT FOLLOWS THE PATH (schema/arrays.ts `ArrayPath`):
//   "vol.bin"           raw, row-major, needs shape (+ dtype, default float32)
//   "w.npy"             numpy .npy v1 / v2 / v3, C or Fortran order, any endianness
//   "data.npz/member"   a member of a zip archive (np.savez / savez_compressed)
//   "vol.zarr/node"     zarr v2 (`.zarray`) or v3 (`zarr.json`) array; compressors null / zlib / gzip
// Elements keep their stored type where a JS typed array exists for it (float32 stays Float32Array, uint8 stays
// Uint8Array; see ArrayData); int64 / uint64 become Float64Array (JS numbers; exact below 2^53), bool becomes Uint8Array.

import type { Dtype } from "@tensatory/schema";
import { NotSupportedError, SpecError } from "../errors";
import { sameType, type ArrayData } from "./ndarray";

/** where a bundle's sidecar files come from; `path` is relative to the bundle document */
export interface ByteSource {
  /** the file's bytes, or null when it does not exist */
  bytes(path: string): Promise<ArrayBuffer | null>;
}

/** a decoded stored array, before any `part` is applied; `data` has the stored element type (see `decodeElements`) */
export interface RawArray {
  readonly shape: readonly number[];
  readonly dtype: Dtype;
  readonly data: ArrayData;
}

/** what the bundle claims about a stored array (checked after decoding; `shape` is REQUIRED for .bin) */
export interface ArrayHint {
  shape?: readonly number[] | undefined;
  dtype?: Dtype | undefined;
}

/** a ByteSource over an in-memory map (unit tests, embedded data) */
export function mapSource(files: Record<string, ArrayBuffer | Uint8Array>): ByteSource {
  return {
    bytes: async (path) => {
      const f = files[path];
      if (f === undefined) return null;
      return f instanceof Uint8Array ? f.slice().buffer as ArrayBuffer : f;
    },
  };
}

/** a ByteSource whose paths are resolved under a directory prefix of another (a member bundle inside a sweep) */
export function rebaseSource(src: ByteSource, dir: string): ByteSource {
  const base = dir === "" || dir.endsWith("/") ? dir : `${dir}/`;
  return { bytes: (path) => src.bytes(base + path) };
}

/*******************************************************/
/* dtypes */

const DTYPE_SIZE: Record<Dtype, number> = {
  float32: 4, float64: 8, int8: 1, uint8: 1, int16: 2, uint16: 2, int32: 4, uint32: 4, int64: 8, uint64: 8, bool: 1,
};
export const dtypeSize = (d: Dtype): number => DTYPE_SIZE[d];

const DTYPES = new Set<string>(Object.keys(DTYPE_SIZE));
export const isDtype = (s: string): s is Dtype => DTYPES.has(s);

/** a numpy descr ("<f4", "|u1", ">i8", "|b1") → dtype + endianness; structured / complex / string descrs are refused */
export function parseDescr(descr: string, at: string[]): { dtype: Dtype; littleEndian: boolean } {
  const m = /^([<>|=])?([fiub])(\d+)$/.exec(descr.trim());
  if (!m) throw new NotSupportedError(`unsupported numpy dtype "${descr}" (only real / integer / bool arrays are supported)`, at);
  const [, order = "|", kind, sizeStr] = m;
  const size = Number(sizeStr);
  const dtype = ((): Dtype | undefined => {
    if (kind === "b") return size === 1 ? "bool" : undefined;
    if (kind === "f") return size === 4 ? "float32" : size === 8 ? "float64" : undefined;
    const name = `${kind === "u" ? "uint" : "int"}${size * 8}`;
    return isDtype(name) ? name : undefined;
  })();
  if (!dtype) throw new NotSupportedError(`unsupported numpy dtype "${descr}"`, at);
  return { dtype, littleEndian: order !== ">" };
}

/** the numpy descr of a dtype (little-endian) — for messages and tests */
export function descrOf(dtype: Dtype): string {
  if (dtype === "bool") return "|b1";
  const size = DTYPE_SIZE[dtype];
  const kind = dtype.startsWith("float") ? "f" : dtype.startsWith("uint") ? "u" : "i";
  return `${size === 1 ? "|" : "<"}${kind}${size}`;
}

/** a zarr v3 data_type name → dtype (v3 uses our names; "int" / "uint" / "float" prefixes with bit widths) */
function parseV3Dtype(name: string, at: string[]): Dtype {
  if (isDtype(name)) return name;
  throw new NotSupportedError(`unsupported zarr data_type "${name}"`, at);
}

const HOST_LITTLE = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

type Bytes = Uint8Array<ArrayBuffer>;

/** the typed array a dtype is stored in */
export function emptyOf(dtype: Dtype, length: number): ArrayData {
  switch (dtype) {
    case "float64": case "int64": case "uint64": return new Float64Array(length);
    case "float32": return new Float32Array(length);
    case "int8": return new Int8Array(length);
    case "uint8": case "bool": return new Uint8Array(length);
    case "int16": return new Int16Array(length);
    case "uint16": return new Uint16Array(length);
    case "int32": return new Int32Array(length);
    case "uint32": return new Uint32Array(length);
  }
}

/**
 * decode `count` elements of `dtype` starting at byte `offset` of `bytes` (a copy, in the host's byte order, of the
 * dtype's own typed array; 64-bit integers as Float64Array since JS has no integer array wider than 32 bits)
 */
export function decodeElements(bytes: Uint8Array, offset: number, count: number, dtype: Dtype, littleEndian: boolean, at: string[]): ArrayData {
  const size = DTYPE_SIZE[dtype];
  const byteLength = count * size;
  if (offset + byteLength > bytes.byteLength)
    throw new SpecError(`array data is too short: ${count} × ${size} bytes needed at offset ${offset}, ${bytes.byteLength - offset} available`, at);
  // an aligned private copy: typed-array views need element alignment, which zip members / chunk buffers do not promise
  const copy = bytes.slice(offset, offset + byteLength);
  if (size > 1 && littleEndian !== HOST_LITTLE) swapBytes(copy, size);
  const buf = copy.buffer as ArrayBuffer;
  switch (dtype) {
    case "float64": return new Float64Array(buf);
    case "float32": return new Float32Array(buf);
    case "int8": return new Int8Array(buf);
    case "uint8": case "bool": return copy;
    case "int16": return new Int16Array(buf);
    case "uint16": return new Uint16Array(buf);
    case "int32": return new Int32Array(buf);
    case "uint32": return new Uint32Array(buf);
    case "int64": return Float64Array.from(new BigInt64Array(buf), Number);
    case "uint64": return Float64Array.from(new BigUint64Array(buf), Number);
  }
}

function swapBytes(a: Uint8Array, size: number): void {
  for (let i = 0; i + size <= a.length; i += size)
    for (let lo = i, hi = i + size - 1; lo < hi; lo++, hi--) { const t = a[lo]!; a[lo] = a[hi]!; a[hi] = t; }
}

/** reorder Fortran-order (column-major) data of `shape` into row-major (same element type) */
export function fortranToC(data: ArrayData, shape: readonly number[]): ArrayData {
  const n = shape.length;
  if (n < 2) return data;
  const out = sameType(data, data.length);
  const fstride = new Array<number>(n);
  for (let d = 0, acc = 1; d < n; d++) { fstride[d] = acc; acc *= shape[d]!; }
  const idx = new Array<number>(n).fill(0);
  let f = 0;
  for (let c = 0; c < data.length; c++) {
    out[c] = data[f]!;
    // odometer over the C-order index; the F offset follows incrementally
    for (let d = n - 1; d >= 0; d--) {
      idx[d]!++; f += fstride[d]!;
      if (idx[d]! < shape[d]!) break;
      f -= idx[d]! * fstride[d]!; idx[d] = 0;
    }
  }
  return out;
}

const product = (xs: readonly number[]): number => xs.reduce((a, b) => a * b, 1);

/*******************************************************/
/* decompression (web streams; deflate-raw for zip members, deflate = zlib-wrapped, gzip) */

async function decompress(bytes: Uint8Array<ArrayBuffer>, format: "deflate" | "deflate-raw" | "gzip"): Promise<Uint8Array<ArrayBuffer>> {
  if (typeof DecompressionStream === "undefined") throw new NotSupportedError("this runtime has no DecompressionStream (compressed arrays need Node ≥ 18 or a modern browser)");
  const stream = new Blob([bytes as BlobPart]).stream().pipeThrough(new DecompressionStream(format));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/*******************************************************/
/* .bin */

function decodeBin(bytes: Uint8Array, hint: ArrayHint, at: string[]): RawArray {
  if (!hint.shape) throw new SpecError("a .bin array needs a declared `shape` (the file has no header)", at);
  const dtype = hint.dtype ?? "float32";
  const n = product(hint.shape);
  if (bytes.byteLength !== n * DTYPE_SIZE[dtype])
    throw new SpecError(`.bin file holds ${bytes.byteLength} bytes but shape [${hint.shape}] of ${dtype} needs ${n * DTYPE_SIZE[dtype]}`, at);
  return { shape: [...hint.shape], dtype, data: decodeElements(bytes, 0, n, dtype, true, at) };
}

/*******************************************************/
/* .npy */

const NPY_MAGIC = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]; // \x93NUMPY

export function decodeNpy(bytes: Uint8Array, at: string[]): RawArray {
  if (bytes.length < 10 || NPY_MAGIC.some((b, i) => bytes[i] !== b)) throw new SpecError("not a .npy file (bad magic)", at);
  const major = bytes[6]!;
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const headerLen = major === 1 ? dv.getUint16(8, true) : dv.getUint32(8, true);
  const headerStart = major === 1 ? 10 : 12;
  const header = new TextDecoder(major >= 3 ? "utf-8" : "latin1").decode(bytes.subarray(headerStart, headerStart + headerLen));
  const descr = /'descr'\s*:\s*'([^']*)'/.exec(header)?.[1];
  const fortran = /'fortran_order'\s*:\s*(True|False)/.exec(header)?.[1];
  const shapeStr = /'shape'\s*:\s*\(([^)]*)\)/.exec(header)?.[1];
  if (descr === undefined || fortran === undefined || shapeStr === undefined) throw new SpecError(`cannot parse .npy header: ${header.trim()}`, at);
  const shape = shapeStr.split(",").map((s) => s.trim()).filter((s) => s.length).map(Number);
  if (shape.some((s) => !Number.isInteger(s) || s < 0)) throw new SpecError(`bad .npy shape (${shapeStr})`, at);
  const { dtype, littleEndian } = parseDescr(descr, at);
  let data = decodeElements(bytes, headerStart + headerLen, product(shape), dtype, littleEndian, at);
  if (fortran === "True") data = fortranToC(data, shape);
  return { shape, dtype, data };
}

/*******************************************************/
/* .npz (zip) */

interface ZipEntry { name: string; method: number; compressedSize: number; size: number; localOffset: number }

/** the central directory of a zip archive (no zip64) */
export function zipEntries(bytes: Uint8Array, at: string[]): ZipEntry[] {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  // end of central directory record: scan back over the (≤ 65535 byte) comment
  let eocd = -1;
  for (let i = bytes.length - 22; i >= Math.max(0, bytes.length - 22 - 65535); i--) if (dv.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
  if (eocd < 0) throw new SpecError("not a zip archive (no end-of-central-directory record)", at);
  const count = dv.getUint16(eocd + 10, true);
  const cdOffset = dv.getUint32(eocd + 16, true);
  if (count === 0xffff || cdOffset === 0xffffffff) throw new NotSupportedError("zip64 archives (> 4 GB or > 65535 members) are not supported", at);
  const entries: ZipEntry[] = [];
  const names = new TextDecoder();
  for (let p = cdOffset, k = 0; k < count; k++) {
    if (dv.getUint32(p, true) !== 0x02014b50) throw new SpecError("corrupt zip central directory", at);
    const method = dv.getUint16(p + 10, true);
    const compressedSize = dv.getUint32(p + 20, true), size = dv.getUint32(p + 24, true);
    const n = dv.getUint16(p + 28, true), e = dv.getUint16(p + 30, true), c = dv.getUint16(p + 32, true);
    const localOffset = dv.getUint32(p + 42, true);
    if (compressedSize === 0xffffffff || size === 0xffffffff || localOffset === 0xffffffff) throw new NotSupportedError("zip64 archives are not supported", at);
    entries.push({ name: names.decode(bytes.subarray(p + 46, p + 46 + n)), method, compressedSize, size, localOffset });
    p += 46 + n + e + c;
  }
  return entries;
}

/** the decompressed bytes of one member */
export async function zipMember(bytes: Bytes, entry: ZipEntry, at: string[]): Promise<Bytes> {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const p = entry.localOffset;
  if (dv.getUint32(p, true) !== 0x04034b50) throw new SpecError(`corrupt zip local header for "${entry.name}"`, at);
  const start = p + 30 + dv.getUint16(p + 26, true) + dv.getUint16(p + 28, true);
  const raw = bytes.subarray(start, start + entry.compressedSize);
  switch (entry.method) {
    case 0: return raw;
    case 8: {
      const out = await decompress(raw, "deflate-raw");
      if (out.length !== entry.size) throw new SpecError(`zip member "${entry.name}" inflated to ${out.length} bytes, expected ${entry.size}`, at);
      return out;
    }
    default: throw new NotSupportedError(`zip compression method ${entry.method} for "${entry.name}" (only stored / deflate)`, at);
  }
}

async function decodeNpzMember(archive: Bytes, member: string, at: string[]): Promise<RawArray> {
  const entries = zipEntries(archive, at);
  const entry = entries.find((e) => e.name === member) ?? entries.find((e) => e.name === `${member}.npy`);
  if (!entry) throw new SpecError(`no member "${member}" in the archive (members: ${entries.map((e) => e.name.replace(/\.npy$/, "")).join(", ")})`, at);
  return decodeNpy(await zipMember(archive, entry, at), at);
}

/*******************************************************/
/* zarr */

interface ZarrMeta {
  shape: number[];
  chunks: number[];
  dtype: Dtype;
  littleEndian: boolean;
  order: "C" | "F";
  fill: number;
  compression: "none" | "zlib" | "gzip";
  /** chunk grid indices → key, relative to the array node */
  key(idx: number[]): string;
}

function fillValue(v: unknown): number {
  if (v === null || v === undefined) return 0;
  if (typeof v === "number") return v;
  if (typeof v === "boolean") return v ? 1 : 0;
  if (v === "NaN") return NaN;
  if (v === "Infinity") return Infinity;
  if (v === "-Infinity") return -Infinity;
  return Number(v);
}

function zarrV2Meta(meta: Record<string, unknown>, at: string[]): ZarrMeta {
  const shape = meta.shape as number[], chunks = meta.chunks as number[];
  if (!Array.isArray(shape) || !Array.isArray(chunks) || shape.length !== chunks.length) throw new SpecError("bad .zarray: shape / chunks", at);
  if (typeof meta.dtype !== "string") throw new NotSupportedError("structured zarr dtypes are not supported", at);
  const { dtype, littleEndian } = parseDescr(meta.dtype, at);
  if (Array.isArray(meta.filters) && meta.filters.length) throw new NotSupportedError(`zarr filters (${meta.filters.map((f) => (f as { id: string }).id).join(", ")}) are not supported`, at);
  const comp = meta.compressor as { id?: string } | null | undefined;
  let compression: ZarrMeta["compression"];
  if (comp === null || comp === undefined) compression = "none";
  else if (comp.id === "zlib") compression = "zlib";
  else if (comp.id === "gzip") compression = "gzip";
  else throw new NotSupportedError(`zarr compressor "${comp.id}" is not supported (write with compressor=None, numcodecs.Zlib or numcodecs.GZip)`, at);
  const sep = (meta.dimension_separator as string | undefined) ?? ".";
  return {
    shape, chunks, dtype, littleEndian, order: meta.order === "F" ? "F" : "C", fill: fillValue(meta.fill_value), compression,
    key: (idx) => (idx.length ? idx.join(sep) : "0"),
  };
}

function zarrV3Meta(meta: Record<string, unknown>, at: string[]): ZarrMeta {
  if (meta.node_type !== "array") throw new SpecError(`zarr node is a ${String(meta.node_type)}, not an array`, at);
  const shape = meta.shape as number[];
  const grid = meta.chunk_grid as { name?: string; configuration?: { chunk_shape?: number[] } } | undefined;
  if (grid?.name !== "regular" || !Array.isArray(grid.configuration?.chunk_shape)) throw new NotSupportedError(`zarr chunk grid "${String(grid?.name)}" is not supported`, at);
  const chunks = grid.configuration!.chunk_shape!;
  if (!Array.isArray(shape) || shape.length !== chunks.length) throw new SpecError("bad zarr.json: shape / chunk_shape", at);
  const dtype = parseV3Dtype(String(meta.data_type), at);
  const enc = meta.chunk_key_encoding as { name?: string; configuration?: { separator?: string } } | undefined;
  const sep = enc?.configuration?.separator ?? (enc?.name === "v2" ? "." : "/");
  const key = enc?.name === "v2" ? (idx: number[]) => (idx.length ? idx.join(sep) : "0") : (idx: number[]) => (idx.length ? `c${sep}${idx.join(sep)}` : "c");
  if (Array.isArray(meta.storage_transformers) && meta.storage_transformers.length) throw new NotSupportedError("zarr storage transformers are not supported", at);
  const codecs = (meta.codecs as { name: string; configuration?: Record<string, unknown> }[] | undefined) ?? [{ name: "bytes" }];
  let littleEndian = true, compression: ZarrMeta["compression"] = "none", order: "C" | "F" = "C";
  let sawBytes = false;
  for (const c of codecs) {
    switch (c.name) {
      case "bytes": case "endian": sawBytes = true; littleEndian = c.configuration?.endian !== "big"; break;
      case "gzip": compression = "gzip"; break;
      case "zlib": case "numcodecs.zlib": compression = "zlib"; break;
      case "transpose": {
        const ord = c.configuration?.order as number[] | string | undefined;
        if (Array.isArray(ord) && ord.every((d, i) => d === i)) break;
        if (Array.isArray(ord) && ord.every((d, i) => d === ord.length - 1 - i)) { order = "F"; break; }
        throw new NotSupportedError(`zarr transpose codec with order ${JSON.stringify(ord)} is not supported`, at);
      }
      default: throw new NotSupportedError(`zarr codec "${c.name}" is not supported (only bytes / gzip / zlib; write with codecs=[BytesCodec(), GzipCodec()])`, at);
    }
  }
  if (!sawBytes) throw new SpecError("zarr.json codecs lack the bytes codec", at);
  return { shape, chunks, dtype, littleEndian, order, fill: fillValue(meta.fill_value), compression, key };
}

/** split "store.zarr/a/b" into the store root and the node path; undefined when the path has no .zarr segment */
export function zarrSplit(path: string): { root: string; node: string } | undefined {
  const segs = path.split("/");
  const i = segs.findIndex((s) => /\.zarr$/i.test(s));
  if (i < 0) return undefined;
  return { root: segs.slice(0, i + 1).join("/"), node: segs.slice(i + 1).join("/") };
}

async function readJson(src: ByteSource, path: string, at: string[]): Promise<Record<string, unknown> | null> {
  const bytes = await src.bytes(path);
  if (bytes === null) return null;
  try { return JSON.parse(new TextDecoder().decode(bytes)) as Record<string, unknown>; }
  catch (e) { throw new SpecError(`"${path}" is not JSON: ${(e as Error).message}`, at); }
}

/** run `f` over `items` with at most `limit` in flight */
async function mapPool<T, R>(items: readonly T[], limit: number, f: (item: T) => Promise<R>): Promise<R[]> {
  const out = new Array<R>(items.length);
  let next = 0;
  const worker = async () => { for (;;) { const i = next++; if (i >= items.length) return; out[i] = await f(items[i]!); } };
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
  return out;
}

async function decodeZarr(src: ByteSource, root: string, node: string, at: string[]): Promise<RawArray> {
  const dir = node ? `${root}/${node}` : root;
  const v2 = await readJson(src, `${dir}/.zarray`, at);
  const meta = v2 ? zarrV2Meta(v2, at) : await (async () => {
    const v3 = await readJson(src, `${dir}/zarr.json`, at);
    if (!v3) throw new SpecError(`"${dir}" is not a zarr array (neither .zarray nor zarr.json found)`, at);
    return zarrV3Meta(v3, at);
  })();
  const { shape, chunks, dtype } = meta;
  const n = shape.length;
  // NaN / infinite fill values only exist for floats; an integer array with such a fill is widened to keep it
  const data = Number.isFinite(meta.fill) || dtype.startsWith("float") ? emptyOf(dtype, product(shape)) : new Float64Array(product(shape));
  if (meta.fill !== 0) data.fill(meta.fill);
  const grid = shape.map((s, d) => Math.ceil(s / chunks[d]!));
  const chunkCount = product(grid);
  const chunkLen = product(chunks);
  const cstrides = new Array<number>(n); // strides of the chunk grid
  for (let d = n - 1, acc = 1; d >= 0; d--) { cstrides[d] = acc; acc *= grid[d]!; }
  const astrides = new Array<number>(n); // strides of the full array
  for (let d = n - 1, acc = 1; d >= 0; d--) { astrides[d] = acc; acc *= shape[d]!; }
  const kstrides = new Array<number>(n); // strides within a chunk (C order)
  for (let d = n - 1, acc = 1; d >= 0; d--) { kstrides[d] = acc; acc *= chunks[d]!; }

  await mapPool(Array.from({ length: chunkCount }, (_, i) => i), 8, async (c) => {
    const idx = new Array<number>(n);
    for (let d = 0, rem = c; d < n; d++) { idx[d] = Math.floor(rem / cstrides[d]!); rem -= idx[d]! * cstrides[d]!; }
    const bytes = await src.bytes(`${dir}/${meta.key(idx)}`);
    if (bytes === null) return; // a missing chunk is all fill_value
    let raw = new Uint8Array(bytes);
    if (meta.compression === "zlib") raw = await decompress(raw, "deflate");
    else if (meta.compression === "gzip") raw = await decompress(raw, "gzip");
    let vals = decodeElements(raw, 0, chunkLen, dtype, meta.littleEndian, [...at, meta.key(idx)]);
    if (meta.order === "F") vals = fortranToC(vals, chunks);
    // copy the in-bounds part of the chunk into the array (edge chunks are padded)
    const origin = idx.map((i, d) => i * chunks[d]!);
    const extent = origin.map((o, d) => Math.min(chunks[d]!, shape[d]! - o));
    if (n === 0) { data[0] = vals[0]!; return; }
    const pos = new Array<number>(n).fill(0);
    const rowLen = extent[n - 1]!;
    for (;;) {
      let a = 0, k = 0;
      for (let d = 0; d < n - 1; d++) { a += (origin[d]! + pos[d]!) * astrides[d]!; k += pos[d]! * kstrides[d]!; }
      a += origin[n - 1]!;
      data.set(vals.subarray(k, k + rowLen), a);
      let d = n - 2;
      for (; d >= 0; d--) { pos[d]!++; if (pos[d]! < extent[d]!) break; pos[d] = 0; }
      if (d < 0) break;
    }
  });
  return { shape: [...shape], dtype, data };
}

/*******************************************************/
/* entry point */

export type ArrayFormat = "bin" | "npy" | "npz" | "zarr";

/** which reader a path selects (by its extension / segments) */
export function formatOf(path: string): ArrayFormat | undefined {
  if (zarrSplit(path)) return "zarr";
  if (/\.npz\/.+/i.test(path)) return "npz";
  if (/\.npy$/i.test(path)) return "npy";
  if (/\.bin$/i.test(path)) return "bin";
  return undefined;
}

/** decode one stored array (before `part`), checking it against the bundle's hint */
export async function loadArray(src: ByteSource, path: string, hint: ArrayHint = {}, at: string[] = []): Promise<RawArray> {
  const format = formatOf(path);
  if (!format) throw new SpecError(`cannot tell the format of array "${path}" (expected .bin, .npy, .npz/member or .zarr/node)`, at);
  let raw: RawArray;
  if (format === "zarr") {
    const { root, node } = zarrSplit(path)!;
    raw = await decodeZarr(src, root, node, at);
  } else if (format === "npz") {
    const m = /^(.*\.npz)\/(.+)$/i.exec(path)!;
    const archive = await src.bytes(m[1]!);
    if (archive === null) throw new SpecError(`file "${m[1]}" not found`, at);
    raw = await decodeNpzMember(new Uint8Array(archive), m[2]!, at);
  } else {
    const bytes = await src.bytes(path);
    if (bytes === null) throw new SpecError(`file "${path}" not found`, at);
    raw = format === "npy" ? decodeNpy(new Uint8Array(bytes), at) : decodeBin(new Uint8Array(bytes), hint, at);
  }
  if (hint.shape && (hint.shape.length !== raw.shape.length || hint.shape.some((s, i) => s !== raw.shape[i])))
    throw new SpecError(`array "${path}" has shape [${raw.shape}] but the bundle declares [${hint.shape}]`, at);
  if (hint.dtype && hint.dtype !== raw.dtype) throw new SpecError(`array "${path}" is ${raw.dtype} but the bundle declares ${hint.dtype}`, at);
  return raw;
}
