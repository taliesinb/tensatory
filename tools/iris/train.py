#!/usr/bin/env python
"""Train a small MLP on iris and bake it into a Tensatory bundle.

Once-off: the bundle (apps/viewer/public/bundles/iris.json) carries the net,
its trained weights, the validation set, and K = 3 orthogonal random
directions around the trained point. The 2D space uses directions 0 and 1,
the 3D space all three. Alongside, a reference file with PyTorch-computed
loss / accuracy at a set of displaced points lets the core tests check that
Tensatory's evaluator agrees with PyTorch.

    ~/projects/loss-landscape/.venv/bin/python tools/iris/train.py

Deterministic (fixed seeds). Data: tools/iris/iris.data (UCI).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUNDLE = ROOT / "apps/viewer/public/bundles/iris.json"
REFERENCE = ROOT / "packages/core/test/fixtures/iris-reference.json"

SEED = 7
HIDDEN = 16
VAL_PER_CLASS = 10
EPOCHS = 400
K = 3  # random directions
SPECIES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]


def load() -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for line in (HERE / "iris.data").read_text().splitlines():
        if not line.strip():
            continue
        *feat, name = line.split(",")
        xs.append([float(f) for f in feat])
        ys.append(SPECIES.index(name))
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.int64)


def split(x: np.ndarray, y: np.ndarray, rng: np.random.Generator):
    tr, va = [], []
    for c in range(3):
        idx = rng.permutation(np.flatnonzero(y == c))
        va += list(idx[:VAL_PER_CLASS])
        tr += list(idx[VAL_PER_CLASS:])
    tr, va = np.array(sorted(tr)), np.array(sorted(va))
    return x[tr], y[tr], x[va], y[va]


def forward(params: dict[str, torch.Tensor], x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    xs = (x - mu) / sigma
    h = torch.relu(xs @ params["W1"] + params["b1"])
    return h @ params["W2"] + params["b2"]


def loss_acc(params, x, y, mu, sigma) -> tuple[float, float]:
    logits = forward(params, x, mu, sigma)
    loss = F.cross_entropy(logits, y)  # mean over the set, matches the bundle's `reduce mean`
    acc = (logits.argmax(-1) == y).double().mean()
    return loss.item(), acc.item()


def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    x, y = load()
    xtr, ytr, xva, yva = split(x, y, rng)
    mu = xtr.mean(0)
    sigma = xtr.std(0)

    # --- train in float32 -------------------------------------------------
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    params = {
        "W1": (torch.randn(4, HIDDEN) * math.sqrt(2 / 4)).requires_grad_(),
        "b1": torch.zeros(HIDDEN).requires_grad_(),
        "W2": (torch.randn(HIDDEN, 3) * math.sqrt(2 / HIDDEN)).requires_grad_(),
        "b2": torch.zeros(3).requires_grad_(),
    }
    opt = torch.optim.Adam(params.values(), lr=0.01, weight_decay=3e-3)
    Xtr, Ytr, Mu, Sig = t(xtr), torch.tensor(ytr), t(mu), t(sigma)
    for epoch in range(EPOCHS):
        opt.zero_grad()
        loss = F.cross_entropy(forward(params, Xtr, Mu, Sig), Ytr)
        loss.backward()
        opt.step()
    train_loss = loss.item()

    # --- everything below in float64, from the EXACT float32 weights -------
    theta = {k: v.detach().double() for k, v in params.items()}
    Xva, Yva, Mu64, Sig64 = torch.tensor(xva), torch.tensor(yva), torch.tensor(mu), torch.tensor(sigma)
    val_loss, val_acc = loss_acc(theta, Xva, Yva, Mu64, Sig64)
    print(f"train loss {train_loss:.4f}   val loss {val_loss:.4f}   val acc {val_acc:.3f}")

    # K orthogonal random directions, each with the norm of theta*
    flat = torch.cat([v.flatten() for v in theta.values()])
    norm = flat.norm().item()
    gen = torch.Generator().manual_seed(SEED)
    dirs_flat = []
    for _ in range(K):
        d = torch.randn(flat.numel(), generator=gen, dtype=torch.float64)
        for prev in dirs_flat:
            d = d - (d @ prev) / (prev @ prev) * prev
        dirs_flat.append(d / d.norm() * norm)

    def unflatten(v: torch.Tensor) -> dict[str, torch.Tensor]:
        out, at = {}, 0
        for k, p in theta.items():
            out[k] = v[at : at + p.numel()].reshape(p.shape)
            at += p.numel()
        return out

    dirs = [unflatten(d) for d in dirs_flat]

    def at(ts: list[float]) -> dict[str, torch.Tensor]:
        return {k: theta[k] + sum(c * dirs[i][k] for i, c in enumerate(ts)) for k in theta}

    # --- reference values ----------------------------------------------------
    points2 = [[0.0, 0.0], [0.5, 0.0], [0.0, -0.5], [0.3, 0.7], [-1.0, 1.0], [0.9, -0.9], [-0.25, -0.75]]
    points3 = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.5], [0.3, -0.7, 0.2], [-1.0, 1.0, -1.0], [0.6, 0.6, 0.6]]
    ref = {"points2": points2, "loss2": [], "acc2": [], "points3": points3, "loss3": [], "acc3": []}
    for p in points2:
        l, a = loss_acc(at(p), Xva, Yva, Mu64, Sig64)
        ref["loss2"].append(l)
        ref["acc2"].append(a)
    for p in points3:
        l, a = loss_acc(at(p), Xva, Yva, Mu64, Sig64)
        ref["loss3"].append(l)
        ref["acc3"].append(a)

    # --- bundle -----------------------------------------------------------------
    def inline(a: torch.Tensor | np.ndarray) -> dict:
        a = np.asarray(a, dtype=np.float64)
        return {"type": "inline", "shape": list(a.shape), "data": [float(v) for v in a.flatten()]}

    def direction(k: int) -> dict:
        return {name: inline(dirs[k][name]) for name in theta}

    box = lambda d: [[-1.0, 1.0]] * d
    field = lambda net, output, domain, name, codomain, description: {
        "kind": "scalar", "name": name, "domain": domain, "codomain": codomain, "description": description,
        "data": {"type": "net", "net": net, "output": output, "box": box(2 if domain == "rand2" else 3)},
    }
    bundle = {
        "tensatory": "0.1",
        "name": "iris MLP",
        "description": (
            f"A 4-{HIDDEN}-3 ReLU MLP trained on iris (120 examples, Adam, {EPOCHS} epochs; train loss {train_loss:.4f}, "
            f"validation loss {val_loss:.4f}, accuracy {val_acc:.3f} on 30 held-out examples). The bundle carries the net, its "
            "trained weights, the validation set and three orthogonal random directions in parameter space, each as long as θ*. "
            "Loss and accuracy on the validation set are evaluated in the browser by the net evaluator: the 2D space is "
            "θ* + t₀ d₀ + t₁ d₁, the 3D space adds d₂. Generated by tools/iris/train.py."
        ),
        "manifolds": {
            "rand2": {"name": "2 random directions", "numDims": 2, "dimNames": ["t₀", "t₁"]},
            "rand3": {"name": "3 random directions", "numDims": 3, "dimNames": ["t₀", "t₁", "t₂"]},
        },
        "nets": {
            "iris": {
                "type": "def",
                "name": "iris MLP",
                "description": "standardize, 4→16 ReLU, 16→3, cross-entropy mean and accuracy over the N examples",
                "inputs": {"x": ["N", 4], "y": ["N"], "W1": [4, HIDDEN], "b1": [HIDDEN], "W2": [HIDDEN, 3], "b2": [3]},
                "arrays": {"mu": inline(mu), "sigma": inline(sigma)},
                "nodes": {
                    "xs": {"op": "div", "vals": [{"op": "sub", "vals": ["x", "mu"]}, "sigma"]},
                    "h": {"op": "relu", "val": {"op": "add", "vals": [{"op": "matmul", "vals": ["xs", "W1"]}, "b1"]}},
                    "logits": {"op": "add", "vals": [{"op": "matmul", "vals": ["h", "W2"]}, "b2"]},
                    "logp": {"op": "logSoftmax", "val": "logits"},
                    "nll": {"op": "negate", "val": {"op": "takeAlong", "val": "logp", "axis": -1, "indices": {"op": "reshape", "val": "y", "shape": ["N", 1]}}},
                    "loss": {"op": "reduce", "fn": "mean", "val": "nll"},
                    "pred": {"op": "argmax", "val": "logits"},
                    "acc": {"op": "reduce", "fn": "mean", "val": {"op": "eq", "vals": ["pred", "y"]}},
                },
                "outputs": {"loss": [], "acc": [], "logits": ["N", 3]},
            },
            "iris_val": {"type": "bind", "net": "iris", "name": "iris MLP on the validation set", "bind": {"x": inline(xva), "y": inline(yva.astype(np.float64))}},
            "iris_star": {"type": "bind", "net": "iris_val", "name": "trained iris MLP (θ*)", "bind": {k: inline(v) for k, v in theta.items()}},
            "iris_rand2": {"type": "displace", "net": "iris_star", "name": "θ* + t₀ d₀ + t₁ d₁", "directions": [direction(0), direction(1)]},
            "iris_rand3": {"type": "displace", "net": "iris_star", "name": "θ* + t₀ d₀ + t₁ d₁ + t₂ d₂", "directions": [direction(0), direction(1), direction(2)]},
        },
        "fields": {
            "loss2": field("iris_rand2", "loss", "rand2", "loss", "celoss", "validation cross-entropy around θ* in the plane of d₀, d₁"),
            "acc2": field("iris_rand2", "acc", "rand2", "accuracy", "fraction", "validation accuracy around θ* in the plane of d₀, d₁"),
            "loss3": field("iris_rand3", "loss", "rand3", "loss", "celoss", "validation cross-entropy around θ* in the space of d₀, d₁, d₂"),
            "acc3": field("iris_rand3", "acc", "rand3", "accuracy", "fraction", "validation accuracy around θ* in the space of d₀, d₁, d₂"),
        },
        "pointSets": {
            "theta2": {"domain": "rand2", "points": [[0.0, 0.0]], "labels": ["θ*"]},
            "theta3": {"domain": "rand3", "points": [[0.0, 0.0, 0.0]], "labels": ["θ*"]},
        },
    }
    BUNDLE.write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")) + "\n")
    REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE.write_text(json.dumps(ref, indent=1) + "\n")
    print(f"wrote {BUNDLE.relative_to(ROOT)} ({BUNDLE.stat().st_size // 1024} kB) and {REFERENCE.relative_to(ROOT)}")
    for p, l, a in zip(points2, ref["loss2"], ref["acc2"]):
        print(f"  2D t={p}: loss {l:.6f} acc {a:.3f}")
    for p, l, a in zip(points3, ref["loss3"], ref["acc3"]):
        print(f"  3D t={p}: loss {l:.6f} acc {a:.3f}")


if __name__ == "__main__":
    main()
