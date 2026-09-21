// Shared test utilities: the environment's half of loading (bytes by path), which core deliberately lacks.

import { readFile } from "node:fs/promises";
import { join } from "node:path";
import type { ByteSource } from "../src";

/** a ByteSource over a directory: bytes by path under `dir`; a missing file (or a path through a file) is null */
export function dirSource(dir: string): ByteSource {
  return {
    bytes: async (path) => {
      try { const b = await readFile(join(dir, path)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer; }
      catch (e) { const code = (e as NodeJS.ErrnoException).code; if (code === "ENOENT" || code === "ENOTDIR" || code === "EISDIR") return null; throw e; }
    },
  };
}

/** a minimal .npy (v1, little-endian, C order) of float32 data, for in-memory fixtures */
export function npyBytes(data: Float32Array, shape: number[]): Uint8Array {
  const dict = `{'descr': '<f4', 'fortran_order': False, 'shape': (${shape.join(", ")}${shape.length === 1 ? "," : ""}), }`;
  const pad = 64 - ((10 + dict.length + 1) % 64);
  const header = dict + " ".repeat(pad) + "\n";
  const out = new Uint8Array(10 + header.length + data.byteLength);
  out.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0]);
  new DataView(out.buffer).setUint16(8, header.length, true);
  for (let i = 0; i < header.length; i++) out[10 + i] = header.charCodeAt(i);
  out.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), 10 + header.length);
  return out;
}

export const jsonBytes = (v: unknown): Uint8Array => new TextEncoder().encode(JSON.stringify(v));
