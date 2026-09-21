# Fixtures for the external-array loaders (core/test/handles.test.ts). Deterministic; run
#   uv run --with numpy --with zarr python packages/core/test/fixtures/handles/make.py
# from the repo root and commit the result. Every array is small; the values are chosen so a
# test can predict them (row-major cell index, or a cell-index expression).
from pathlib import Path
import json, shutil
import numpy as np

HERE = Path(__file__).parent
for p in HERE.iterdir():
    if p.name not in ("make.py", "bundle.json"):
        shutil.rmtree(p) if p.is_dir() else p.unlink()

def ramp(shape, dtype):
    return np.arange(int(np.prod(shape)), dtype=np.float64).reshape(shape).astype(dtype)

# ---- .npy: dtypes, orders, endianness, versions
np.save(HERE / "f4.npy", ramp((2, 3), "<f4"))
np.save(HERE / "f8.npy", ramp((2, 3), "<f8") / 8)
np.save(HERE / "i4.npy", ramp((5,), "<i4") - 2)
np.save(HERE / "i8.npy", (ramp((3,), "<i8") - 1) * 2**40)
np.save(HERE / "u1.npy", ramp((2, 2), "u1"))
np.save(HERE / "u2.npy", ramp((2, 2), "<u2") * 300)
np.save(HERE / "bool.npy", np.array([[True, False], [False, True]]))
np.save(HERE / "bigendian.npy", ramp((2, 3), ">f4"))
np.save(HERE / "fortran.npy", np.asfortranarray(ramp((2, 3, 4), "<f8")))
np.save(HERE / "scalar.npy", np.float32(42.5))
# v2 header (forced): numpy writes v2 when the header exceeds 65535 bytes; use the format API
from numpy.lib import format as npf
with open(HERE / "v2.npy", "wb") as f:
    npf.write_array(f, ramp((3,), "<f4"), version=(2, 0))
# a large volume: (4, 5, 6, 2) channel-interleaved, cell = flat index
np.save(HERE / "vol.npy", ramp((4, 5, 6, 2), "<f4"))

# ---- .bin: raw row-major float32 of the same volume, and a float64 vector
ramp((4, 5, 6, 2), "<f4").tofile(HERE / "vol.bin")
np.array([1.5, -2.5, 3.5], "<f8").tofile(HERE / "vec.bin")

# ---- .npz: stored and deflated, nested member names
np.savez(HERE / "stored.npz", a=ramp((2, 2), "<f4"), **{"nested/b": ramp((3,), "<i4")})
np.savez_compressed(HERE / "deflated.npz", a=ramp((10, 10), "<f8"), w=ramp((3, 2), "<f4"))

# ---- zarr
import zarr, numcodecs
from numcodecs import Zlib, GZip

vol = ramp((4, 5, 6, 2), "<f4")
# v2, uncompressed, "." separator, edge chunks (chunk 3 over size 4/5/6), a group with two arrays
g2 = zarr.open_group(str(HERE / "v2.zarr"), mode="w", zarr_format=2)
g2.create_array("vol", data=vol, chunks=(3, 3, 4, 2), compressors=None)
g2.create_array("zlib", data=ramp((7, 5), "<f8"), chunks=(4, 4), compressors=Zlib(level=6))
g2.create_array("gzip", data=ramp((6,), "<i4"), chunks=(4,), compressors=GZip(level=5))
g2.create_array("fill", shape=(4, 4), chunks=(2, 2), dtype="<f4", fill_value=7.5, compressors=None)  # never written: all fill
g2.create_array("fortran", data=np.asfortranarray(ramp((3, 4), "<f8")), chunks=(3, 4), order="F", compressors=None)
g2.create_array("scalar", data=np.float64(3.25), chunks=(), compressors=None) if hasattr(zarr, "create_array") else None
# v2 with "/" dimension separator, as a root array
zarr.create_array(str(HERE / "v2sep.zarr"), data=ramp((5, 3), "<f4"), chunks=(2, 2), zarr_format=2, compressors=None,
                  chunk_key_encoding={"name": "v2", "separator": "/"})
# v2 with zstd (zarr-python 3's default, also for v2): refused
zarr.create_array(str(HERE / "zstd2.zarr"), data=ramp((4,), "<f4"), chunks=(4,), zarr_format=2)

# v3: bytes + gzip codecs, default "c/" key encoding, edge chunks
from zarr.codecs import BytesCodec, GzipCodec
g3 = zarr.open_group(str(HERE / "v3.zarr"), mode="w", zarr_format=3)
g3.create_array("vol", data=vol, chunks=(3, 3, 4, 2), compressors=[GzipCodec(level=4)], serializer=BytesCodec())
g3.create_array("raw", data=ramp((5,), "<i8") - 2, chunks=(2,), compressors=[], serializer=BytesCodec(endian="big"))
g3.create_array("fill", shape=(3,), chunks=(3,), dtype="float32", fill_value=float("nan"), compressors=[])
# v3 with the default (zstd) codec: refused
zarr.create_array(str(HERE / "zstd.zarr"), data=ramp((4,), "<f4"), chunks=(4,), zarr_format=3)

# a listing the test can compare against
listing = sorted(str(p.relative_to(HERE)) for p in HERE.rglob("*") if p.is_file() and p.name not in ("make.py", "bundle.json", "listing.json"))
(HERE / "listing.json").write_text(json.dumps(listing, indent=1) + "\n")
print("\n".join(listing))
