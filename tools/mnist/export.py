#!/usr/bin/env python
"""Export the loss-landscape prototype's trained MNIST MLP as a LIVE Tensatory bundle.

Reads ~/projects/loss-landscape/out/mnist_mlp.pt (theta* + the SGD trajectory, from landscape.py) and
viewer/data/mnist_mlp_pca.{json,bin} (the 64^3 PCA volume volume.py evaluated with PyTorch), and writes
apps/viewer/public/bundles/mnist-mlp/:

  bundle.json   the net (784-256-256-10 ReLU MLP, cross-entropy over N examples), theta* bound, displaced along
                the top-3 PCA directions of the trajectory; loss / accuracy fields on a 2D and a 3D PCA space,
                evaluated IN THE BROWSER by the net evaluator; the projected trajectory and theta* as point sets;
                the prototype's own sampled volume (subsampled) as reference fields in the 3D space
  arrays.npz    theta* (weights transposed to [in, out]), the 3 PCA directions likewise, the eval set as uint8
                pixels (the normalization is in the net), labels, and the same for the first 256 examples
  vol.npy       the prototype's volume, every other sample of 64^3 -> 32^3, (z, y, x, c) float32 as stored

and packages/core/test/fixtures/mnist-mlp-reference.json: PyTorch's loss / accuracy at theta* and at displaced
points, for core/test/mnist.test.ts to check the evaluator against.

    ~/projects/loss-landscape/.venv/bin/python tools/mnist/export.py

Deterministic: everything is read from files or recomputed with the prototype's fixed seeds (the eval subset is
torch.randperm(60000, seed 0)[:1024], as in landscape.get_data).
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTO = Path.home() / "projects/loss-landscape"
OUT = ROOT / "apps/viewer/public/bundles/mnist-mlp"
REFERENCE = ROOT / "packages/core/test/fixtures/mnist-mlp-reference.json"

sys.path.insert(0, str(PROTO))
from landscape import get_data, pca_directions  # noqa: E402  (the prototype's own data pipeline and PCA)

MEAN, STD = 0.1307, 0.3081  # landscape.DATASETS["mnist"]
EVAL_N = 1024  # what volume.py used for the volumes
SMALL_N = 256  # a cheaper bind for interactive use
SIZES = [("W1t", (784, 256)), ("b1", (256,)), ("W2t", (256, 256)), ("b2", (256,)), ("W3t", (256, 10)), ("b3", (10,))]


def unflatten(flat: torch.Tensor) -> dict[str, torch.Tensor]:
    """torch's parameter order (Linear.weight [out, in], Linear.bias) -> our transposed names [in, out]"""
    out, i = {}, 0
    for (name, shape), torch_shape in zip(SIZES, [(256, 784), (256,), (256, 256), (256,), (10, 256), (10,)]):
        n = int(np.prod(torch_shape))
        t = flat[i : i + n].view(torch_shape)
        out[name] = t.T.contiguous() if name.startswith("W") else t
        i += n
    assert i == flat.numel()
    return out


def forward(p: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    xs = (x / 255.0 - MEAN) / STD
    h1 = F.relu(xs @ p["W1t"] + p["b1"])
    h2 = F.relu(h1 @ p["W2t"] + p["b2"])
    return h2 @ p["W3t"] + p["b3"]


def loss_acc(p, x, y) -> tuple[float, float]:
    logits = forward(p, x)
    return F.cross_entropy(logits, y).item(), (logits.argmax(1) == y).double().mean().item()


def main() -> None:
    torch.set_default_dtype(torch.float64)
    ck = torch.load(PROTO / "out/mnist_mlp.pt", map_location="cpu")
    traj = [t.double() for t in ck["traj"]]
    theta = traj[-1]
    print(f"theta*: {theta.numel():,} params, {len(traj)} snapshots")

    # the eval subset exactly as the prototype drew it (normalized tensors -> back to uint8 pixels)
    _, _, (X, y), _ = get_data("mnist", root=str(PROTO / "data"), eval_n=EVAL_N)
    pixels = torch.round((X.double() * STD + MEAN) * 255.0).clamp(0, 255).view(EVAL_N, 784)
    assert torch.allclose((pixels / 255.0 - MEAN) / STD, X.double().view(EVAL_N, 784), atol=1e-6)
    x_u8 = pixels.to(torch.uint8).numpy()
    y_u8 = y.to(torch.uint8).numpy()
    xd, yd = pixels, y

    dirs, coords, var = pca_directions(traj, k=3)
    dirs = [d.double() for d in dirs]
    print(f"PCA explained variance: {', '.join(f'{v:.1%}' for v in var)}")
    meta = json.loads((PROTO / "viewer/data/mnist_mlp_pca.json").read_text())
    axes = meta["axes"]  # the volume's box, so the live fields and the sampled reference share it
    assert np.allclose(np.array(meta["trajectory"]), coords.numpy(), atol=1e-4), "trajectory projection differs from the volume's"
    assert np.allclose(meta["explained_variance"], var, atol=1e-6)

    p_star = unflatten(theta)
    p_dirs = [unflatten(d) for d in dirs]

    # ---- arrays.npz
    OUT.mkdir(parents=True, exist_ok=True)
    members: dict[str, np.ndarray] = {}
    for k, v in p_star.items():
        members[f"theta/{k}"] = v.numpy().astype(np.float32)
    for i, pd in enumerate(p_dirs):
        for k, v in pd.items():
            members[f"pca/d{i}/{k}"] = v.numpy().astype(np.float32)
    members["data/x"] = x_u8
    members["data/y"] = y_u8
    members["data/x256"] = x_u8[:SMALL_N]
    members["data/y256"] = y_u8[:SMALL_N]
    np.savez_compressed(OUT / "arrays.npz", **members)
    with zipfile.ZipFile(OUT / "arrays.npz") as z:
        names = sorted(n[:-4] for n in z.namelist())
    print(f"arrays.npz: {(OUT / 'arrays.npz').stat().st_size / 1e6:.1f} MB, {len(names)} members")

    # ---- vol.npy: the prototype's 64^3 volume, every other sample (the last sample of 64 is index 63; keep 0..62)
    vol = np.fromfile(PROTO / "viewer/data/mnist_mlp_pca.bin", dtype="<f4").reshape(64, 64, 64, 2)  # (z, y, x, c)
    sub = vol[::2, ::2, ::2, :]
    np.save(OUT / "vol.npy", np.ascontiguousarray(sub))
    sub_axes = [[a, a + (b - a) * 62 / 63] for a, b in axes]
    print(f"vol.npy: {sub.shape} float32, {(OUT / 'vol.npy').stat().st_size // 1024} kB")

    # ---- float32 weights are what the bundle carries: the reference is PyTorch on THOSE (float64 arithmetic)
    p32 = {k: torch.tensor(members[f"theta/{k}"], dtype=torch.float64) for k, _ in SIZES}
    d32 = [{k: torch.tensor(members[f"pca/d{i}/{k}"], dtype=torch.float64) for k, _ in SIZES} for i in range(3)]

    def at(ts: list[float]) -> dict[str, torch.Tensor]:
        return {k: p32[k] + sum(t * d32[i][k] for i, t in enumerate(ts)) for k, _ in SIZES}

    rng = np.random.default_rng(3)
    lo = np.array([a for a, _ in axes]); hi = np.array([b for _, b in axes])
    points3 = [[0.0, 0.0, 0.0]] + (lo + rng.random((5, 3)) * (hi - lo)).round(4).tolist()
    points2 = [[0.0, 0.0]] + (lo[:2] + rng.random((4, 2)) * (hi[:2] - lo[:2])).round(4).tolist()
    ref = {
        "eval_n": EVAL_N, "small_n": SMALL_N, "axes": axes, "sub_axes": sub_axes, "explained_variance": var,
        "points3": points3, "points2": points2,
        "full3": [dict(zip(("loss", "acc"), loss_acc(at(t), xd, yd))) for t in points3],
        "full2": [dict(zip(("loss", "acc"), loss_acc(at(t), xd, yd))) for t in points2],
        "small3": [dict(zip(("loss", "acc"), loss_acc(at(t), xd[:SMALL_N], yd[:SMALL_N]))) for t in points3],
        "small2": [dict(zip(("loss", "acc"), loss_acc(at(t), xd[:SMALL_N], yd[:SMALL_N]))) for t in points2],
    }
    REFERENCE.write_text(json.dumps(ref, indent=1) + "\n")
    star = ref["full3"][0]
    print(f"theta* on the {EVAL_N} eval examples: loss {star['loss']:.6f} (log10 {np.log10(star['loss']):.4f}; the volume's centre says "
          f"{meta['center_log10_loss']:.4f}), acc {star['acc']:.4f} (volume: {meta['center_accuracy']:.4f})")

    # ---- bundle.json
    def net_def() -> dict:
        return {
            "type": "def",
            "name": "MNIST MLP",
            "details": "pixels/255 standardized (mean 0.1307, std 0.3081), 784→256 ReLU, 256→256 ReLU, 256→10; over the N examples: "
                       "cross-entropy mean (loss) and accuracy (acc). Weights are stored [in, out] (PyTorch's transposed).",
            "inputs": {"x": ["N", 784], "y": ["N"], **{k: list(s) for k, s in SIZES}},
            "nodes": {
                "xs": {"op": "div", "vals": [{"op": "sub", "vals": [{"op": "div", "vals": ["x", 255.0]}, MEAN]}, STD]},
                "h1": {"op": "relu", "val": {"op": "add", "vals": [{"op": "matmul", "vals": ["xs", "W1t"]}, "b1"]}},
                "h2": {"op": "relu", "val": {"op": "add", "vals": [{"op": "matmul", "vals": ["h1", "W2t"]}, "b2"]}},
                "logits": {"op": "add", "vals": [{"op": "matmul", "vals": ["h2", "W3t"]}, "b3"]},
                "logp": {"op": "logSoftmax", "val": "logits"},
                "nll": {"op": "negate", "val": {"op": "takeAlong", "val": "logp", "axis": -1, "indices": {"op": "reshape", "val": "y", "shape": ["N", 1]}}},
                "loss": {"op": "reduce", "fn": "mean", "val": "nll"},
                "pred": {"op": "argmax", "val": "logits"},
                "acc": {"op": "reduce", "fn": "mean", "val": {"op": "eq", "vals": ["pred", "y"]}},
            },
            "outputs": {"loss": [], "acc": []},
        }

    def direction(i: int) -> dict:
        return {"name": f"pc{i + 1}", "arrays": {k: f"arrays.npz/pca/d{i}/{k}" for k, _ in SIZES}}

    def displaced(base: str, k: int, name: str) -> dict:
        return {"type": "displace", "net": base, "name": name, "directions": [direction(i) for i in range(k)]}

    def field(net: str, output: str, domain: str, name: str, codomain: str, summary: str, box) -> dict:
        return {"kind": "scalar", "name": name, "domain": domain, "codomain": codomain, "summary": summary,
                "data": {"type": "net", "net": net, "output": output, "box": box}}

    def log10_field(of: str, domain: str, name: str, summary: str) -> dict:
        return {"kind": "scalar", "name": name, "domain": domain, "codomain": "lin", "summary": summary,
                "data": {"type": "pointwise", "expr": {"op": "log10", "val": "l"}, "scalars": {"l": of}}}

    def sampled(c: int, name: str, codomain: str, summary: str) -> dict:
        return {"kind": "scalar", "name": name, "domain": "pca3", "codomain": codomain, "summary": summary,
                "data": {"type": "dense", "box": sub_axes,
                         "samples": {"type": "handle", "path": "vol.npy", "shape": [32, 32, 32, 2], "part": [None, None, None, c], "axes": [2, 1, 0], "dtype": "float32"}}}

    box3 = axes
    box2 = axes[:2]
    traj = coords.numpy()
    bundle = {
        "tensatory": "0.1",
        "name": "MNIST MLP: live PCA landscape",
        "summary": f"The loss-landscape prototype's trained 784-256-256-10 MLP ({theta.numel():,} parameters) evaluated IN THE BROWSER by the net "
                   f"evaluator: cross-entropy and accuracy on {EVAL_N} (or {SMALL_N}) training examples around θ* along the top-3 PCA directions of "
                   f"its SGD trajectory; weights, directions and data as .npz members; the prototype's PyTorch-sampled volume alongside for comparison.",
        "details": (
            f"From ~/projects/loss-landscape/out/mnist_mlp.pt (landscape.py: SGD with momentum, 3 epochs, {len(traj)} snapshots). The PCA directions "
            f"are the top right-singular vectors of θ_t − θ* over the trajectory (explained variance {', '.join(f'{v:.1%}' for v in var)}); the "
            f"trajectory is projected onto them and θ* sits at the origin. The box is the one volume.py used for the 64³ volume "
            f"(viewer/data/mnist_mlp_pca), whose every-other sample is vol.npy here, so `sampled/…` (PyTorch, N = {EVAL_N}) and the live "
            f"`loss` (the net evaluator, same {EVAL_N} examples) should agree to float32. `fast/…` uses the first {SMALL_N} examples. "
            f"Weights are float32 .npz members ([in, out]); the eval set is uint8 pixels, normalized inside the net. Generated by tools/mnist/export.py."
        ),
        "manifolds": {
            "pca2": {"name": "PCA plane", "numDims": 2, "dimNames": ["pc1", "pc2"], "dimWeights": var[:2],
                     "summary": "θ* + t₁ pc1 + t₂ pc2: the top-2 PCA directions of the SGD trajectory (θ* at 0)"},
            "pca3": {"name": "PCA space", "numDims": 3, "dimNames": ["pc1", "pc2", "pc3"], "dimWeights": var,
                     "summary": "θ* + t₁ pc1 + t₂ pc2 + t₃ pc3: the top-3 PCA directions (θ* at 0); also the prototype's sampled volume"},
        },
        "nets": {
            "mlp": net_def(),
            "mlp_eval": {"type": "bind", "net": "mlp", "name": f"MLP on {EVAL_N} training examples", "bind": {"x": "arrays.npz/data/x", "y": "arrays.npz/data/y"}},
            "mlp_star": {"type": "bind", "net": "mlp_eval", "name": "trained MLP (θ*)", "bind": {k: f"arrays.npz/theta/{k}" for k, _ in SIZES}},
            "mlp_pca2": displaced("mlp_star", 2, "θ* + t₁ pc1 + t₂ pc2"),
            "mlp_pca3": displaced("mlp_star", 3, "θ* + t₁ pc1 + t₂ pc2 + t₃ pc3"),
            "mlp_eval256": {"type": "bind", "net": "mlp", "name": f"MLP on the first {SMALL_N} eval examples", "bind": {"x": "arrays.npz/data/x256", "y": "arrays.npz/data/y256"}},
            "mlp_star256": {"type": "bind", "net": "mlp_eval256", "name": f"trained MLP (θ*), {SMALL_N} examples", "bind": {k: f"arrays.npz/theta/{k}" for k, _ in SIZES}},
            "mlp_pca2_256": displaced("mlp_star256", 2, f"θ* + t₁ pc1 + t₂ pc2 ({SMALL_N} examples)"),
            "mlp_pca3_256": displaced("mlp_star256", 3, f"θ* + t₁ pc1 + t₂ pc2 + t₃ pc3 ({SMALL_N} examples)"),
        },
        "fields": {
            "loss2": field("mlp_pca2", "loss", "pca2", "loss", "celoss", f"cross-entropy on {EVAL_N} training examples, live", box2),
            "acc2": field("mlp_pca2", "acc", "pca2", "accuracy", "fraction", f"accuracy on {EVAL_N} training examples, live", box2),
            "log10loss2": log10_field("loss2", "pca2", "log10 loss", "log₁₀ of the live loss (the prototype's channel)"),
            "fastLoss2": field("mlp_pca2_256", "loss", "pca2", "fast/loss", "celoss", f"cross-entropy on the first {SMALL_N} examples (4× cheaper)", box2),
            "fastAcc2": field("mlp_pca2_256", "acc", "pca2", "fast/accuracy", "fraction", f"accuracy on the first {SMALL_N} examples", box2),
            "loss3": field("mlp_pca3", "loss", "pca3", "loss", "celoss", f"cross-entropy on {EVAL_N} training examples, live", box3),
            "acc3": field("mlp_pca3", "acc", "pca3", "accuracy", "fraction", f"accuracy on {EVAL_N} training examples, live", box3),
            "log10loss3": log10_field("loss3", "pca3", "log10 loss", "log₁₀ of the live loss (the prototype's channel)"),
            "fastLoss3": field("mlp_pca3_256", "loss", "pca3", "fast/loss", "celoss", f"cross-entropy on the first {SMALL_N} examples (4× cheaper)", box3),
            "fastAcc3": field("mlp_pca3_256", "acc", "pca3", "fast/accuracy", "fraction", f"accuracy on the first {SMALL_N} examples", box3),
            "sampledLog10Loss": sampled(0, "sampled/log10 loss", "lin", f"the prototype's PyTorch-sampled log₁₀ loss (64³ → 32³, N = {EVAL_N})"),
            "sampledAcc": sampled(1, "sampled/accuracy", "fraction", f"the prototype's PyTorch-sampled accuracy (64³ → 32³, N = {EVAL_N})"),
        },
        "pointSets": {
            "traj2": {"domain": "pca2", "points": traj[:, :2].round(5).tolist(), "ordered": True, "name": "SGD trajectory"},
            "traj3": {"domain": "pca3", "points": traj.round(5).tolist(), "ordered": True, "name": "SGD trajectory"},
            "theta2": {"domain": "pca2", "points": [[0.0, 0.0]], "labels": ["θ*"]},
            "theta3": {"domain": "pca3", "points": [[0.0, 0.0, 0.0]], "labels": ["θ*"]},
        },
    }
    (OUT / "bundle.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}/bundle.json and {REFERENCE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
