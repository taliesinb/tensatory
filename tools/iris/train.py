#!/usr/bin/env python
"""Train a small MLP on iris and bake it into a Tensatory bundle.

Once-off: the bundle (apps/viewer/public/bundles/iris.json) carries the net,
its trained weights, the validation set, and two kinds of directions around
the trained point, on the validation set (30 held out) AND the training set
(120): validation loss / accuracy, training loss / accuracy, the training
objective Adam actually minimized (loss + wd/2 ‖θ‖², weight decay 3e-3) and
the training loss per class. Directions:
  * iris_rnd2 / iris_rnd3 (the viewer's fields): `random` gaussian directions
    drawn from seeds in the browser, each normalized to the length of theta*
    (`norm: "origin"`), with a Controls-pane row each (reseed / scale);
  * iris_rand2 / iris_rand3 (no fields): K = 3 fixed, inline, orthogonal
    random directions, along which the reference file with PyTorch-computed
    loss / accuracy at a set of displaced points is evaluated, so the core
    tests can check that Tensatory's evaluator agrees with PyTorch exactly.
The 2D space uses directions 0 and 1, the 3D space all three.

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


WD = 3e-3  # Adam weight decay: the gradient gets wd·θ, i.e. the objective is loss + wd/2 ‖θ‖²


def loss_acc(params, x, y, mu, sigma) -> tuple[float, float]:
    loss, acc, *_ = metrics(params, x, y, mu, sigma)
    return loss, acc


def metrics(params, x, y, mu, sigma) -> tuple[float, float, float, list[float]]:
    """loss, accuracy, regularized objective, per-class loss — exactly the bundle's `iris` net outputs"""
    logits = forward(params, x, mu, sigma)
    nll = F.cross_entropy(logits, y, reduction="none")
    loss = nll.mean()  # matches the bundle's `reduce mean`
    acc = (logits.argmax(-1) == y).double().mean()
    reg = 0.5 * WD * sum((p * p).sum() for p in params.values())
    per_class = [nll[y == c].mean().item() for c in range(3)]
    return loss.item(), acc.item(), (loss + reg).item(), per_class


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
    opt = torch.optim.Adam(params.values(), lr=0.01, weight_decay=WD)
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
    Xtr64, Ytr64 = torch.tensor(xtr), torch.tensor(ytr)
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
    ref: dict = {"points2": points2, "loss2": [], "acc2": [], "points3": points3, "loss3": [], "acc3": [], "train2": [], "train3": []}
    for p in points2:
        l, a = loss_acc(at(p), Xva, Yva, Mu64, Sig64)
        ref["loss2"].append(l)
        ref["acc2"].append(a)
        tl, ta, to, tc = metrics(at(p), Xtr64, Ytr64, Mu64, Sig64)
        ref["train2"].append({"loss": tl, "acc": ta, "obj": to, "perClass": tc})
    for p in points3:
        l, a = loss_acc(at(p), Xva, Yva, Mu64, Sig64)
        ref["loss3"].append(l)
        ref["acc3"].append(a)
        tl, ta, to, tc = metrics(at(p), Xtr64, Ytr64, Mu64, Sig64)
        ref["train3"].append({"loss": tl, "acc": ta, "obj": to, "perClass": tc})

    # --- bundle -----------------------------------------------------------------
    def inline(a: torch.Tensor | np.ndarray) -> dict:
        a = np.asarray(a, dtype=np.float64)
        return {"type": "inline", "shape": list(a.shape), "data": [float(v) for v in a.flatten()]}

    def direction(k: int) -> dict:
        return {"arrays": {name: inline(dirs[k][name]) for name in theta}}

    SUB = "₀₁₂"

    def random_direction(k: int) -> dict:
        return {
            "name": f"d{SUB[k]}", "norm": "origin", "widget": {"id": f"d{k}"},
            "arrays": {name: {"type": "random", "shape": list(p.shape), "dist": {"type": "gaussian", "seed": f"d{k}/{name}"}} for name, p in theta.items()},
        }

    box = lambda d: [[-1.0, 1.0]] * d
    field = lambda net, output, domain, name, codomain, details: {
        "kind": "scalar", "name": name, "domain": domain, "codomain": codomain, "details": details,
        "data": {"type": "net", "net": net, "output": output, "box": box(2 if domain == "rand2" else 3)},
    }
    bundle = {
        "tensatory": "0.1",
        "name": "iris MLP",
        "summary": "A 4-16-3 ReLU MLP trained on iris; validation and training loss / accuracy / per-class loss landscapes around θ* along 2 or 3 random directions (reseed / rescale in Controls).",
        "details": (
            f"A 4-{HIDDEN}-3 ReLU MLP trained on iris (120 examples, Adam, {EPOCHS} epochs; train loss {train_loss:.4f}, "
            f"validation loss {val_loss:.4f}, accuracy {val_acc:.3f} on 30 held-out examples). The bundle carries the net, its "
            "trained weights, the validation set and random gaussian directions in parameter space, each as long as θ* (drawn from seeds "
            "in the browser: reseed / rescale them in the Controls pane; the fixed directions of the PyTorch reference are kept as "
            "iris_rand2 / iris_rand3). "
            "Loss and accuracy on the validation set are evaluated in the browser by the net evaluator: the 2D space is "
            "θ* + t₀ d₀ + t₁ d₁, the 3D space adds d₂. Generated by tools/iris/train.py."
        ),
        "manifolds": {
            "rand2": {"name": "2 random directions", "numDims": 2, "dimNames": ["t₀", "t₁"], "summary": "θ* + t₀ d₀ + t₁ d₁: two random gaussian directions, each as long as θ*; validation fields and train/… fields"},
            "rand3": {"name": "3 random directions", "numDims": 3, "dimNames": ["t₀", "t₁", "t₂"], "summary": "θ* + Σ tᵢ dᵢ along three random directions (d₀, d₁ shared with the 2D space): loss / accuracy volumes"},
        },
        "nets": {
            "iris": {
                "type": "def",
                "name": "iris MLP",
                "details": "standardize, 4→16 ReLU, 16→3; over the N examples: cross-entropy mean (loss), accuracy (acc), loss + wd/2 ‖θ‖² (obj, the training objective) and the mean loss of each class (loss0..2)",
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
                    # the objective Adam minimized: weight_decay adds wd·θ to the gradient, i.e. loss + wd/2 ‖θ‖²
                    "reg": {"op": "mul", "vals": [WD / 2, {"op": "add", "vals": [{"op": "reduce", "fn": "sum", "val": {"op": "square", "val": w}} for w in ("W1", "b1", "W2", "b2")]}]},
                    "obj": {"op": "add", "vals": ["loss", "reg"]},
                    # per-class mean loss: mask the examples of class c (nll is [N, 1] from takeAlong: flatten it first)
                    "nllv": {"op": "reshape", "val": "nll", "shape": ["N"]},
                    **{f"mask{c}": {"op": "eq", "vals": ["y", float(c)]} for c in range(3)},
                    **{f"loss{c}": {"op": "div", "vals": [{"op": "reduce", "fn": "sum", "val": {"op": "mul", "vals": ["nllv", f"mask{c}"]}}, {"op": "reduce", "fn": "sum", "val": f"mask{c}"}]} for c in range(3)},
                },
                "outputs": {"loss": [], "acc": [], "obj": [], "loss0": [], "loss1": [], "loss2": [], "logits": ["N", 3]},
            },
            "iris_val": {"type": "bind", "net": "iris", "name": "iris MLP on the validation set", "bind": {"x": inline(xva), "y": inline(yva.astype(np.float64))}},
            "iris_star": {"type": "bind", "net": "iris_val", "name": "trained iris MLP (θ*)", "bind": {k: inline(v) for k, v in theta.items()}},
            "iris_trn": {"type": "bind", "net": "iris", "name": "iris MLP on the training set", "bind": {"x": inline(xtr), "y": inline(ytr.astype(np.float64))}},
            "iris_trn_star": {"type": "bind", "net": "iris_trn", "name": "trained iris MLP (θ*) on the training set", "bind": {k: inline(v) for k, v in theta.items()}},
            "iris_trn_rand2": {"type": "displace", "net": "iris_trn_star", "name": "training set, fixed reference directions (2)", "directions": [direction(0), direction(1)]},
            "iris_trn_rand3": {"type": "displace", "net": "iris_trn_star", "name": "training set, fixed reference directions (3)", "directions": [direction(0), direction(1), direction(2)]},
            "iris_trn_rnd2": {
                "type": "displace", "net": "iris_trn_star", "name": "training set: θ* + t₀ d₀ + t₁ d₁",
                "details": "the same random directions as iris_rnd2 (same seeds, same Controls rows), around θ* on the training set",
                "directions": [random_direction(0), random_direction(1)],
            },
            "iris_trn_rnd3": {
                "type": "displace", "net": "iris_trn_star", "name": "training set: θ* + t₀ d₀ + t₁ d₁ + t₂ d₂",
                "details": "the same random directions as iris_rnd3, around θ* on the training set",
                "directions": [random_direction(0), random_direction(1), random_direction(2)],
            },
            "iris_rand2": {
                "type": "displace", "net": "iris_star", "name": "θ* + t₀ d₀ + t₁ d₁ (fixed reference directions)",
                "details": "the three orthogonal random directions the PyTorch reference (core/test/fixtures/iris-reference.json) was computed along; inline, so tests stay exact",
                "directions": [direction(0), direction(1)],
            },
            "iris_rand3": {
                "type": "displace", "net": "iris_star", "name": "θ* + t₀ d₀ + t₁ d₁ + t₂ d₂ (fixed reference directions)",
                "details": "the three orthogonal random directions the PyTorch reference (core/test/fixtures/iris-reference.json) was computed along; inline, so tests stay exact",
                "directions": [direction(0), direction(1), direction(2)],
            },
            "iris_rnd2": {
                "type": "displace", "net": "iris_star", "name": "θ* + t₀ d₀ + t₁ d₁",
                "details": "two random gaussian directions in parameter space, each as long as θ*; reseed / rescale them in the Controls pane",
                "directions": [random_direction(0), random_direction(1)],
            },
            "iris_rnd3": {
                "type": "displace", "net": "iris_star", "name": "θ* + t₀ d₀ + t₁ d₁ + t₂ d₂",
                "details": "three random gaussian directions in parameter space, each as long as θ*; d₀ and d₁ are the same draws as in the 2D space",
                "directions": [random_direction(0), random_direction(1), random_direction(2)],
            },
        },
        "fields": {
            **{
                f"{fid}{d}": field(f"{net}{d}", out, f"rand{d}", name, cd, f"{desc} around θ* in the {'plane' if d == 2 else 'space'} of d₀, d₁{', d₂' if d == 3 else ''}")
                for d in (2, 3)
                for fid, net, out, name, cd, desc in [
                    ("loss", "iris_rnd", "loss", "loss", "celoss", "validation cross-entropy"),
                    ("acc", "iris_rnd", "acc", "accuracy", "fraction", "validation accuracy"),
                    # names are paths: the mappings table shows "train" as a collapsible heading with these beneath
                    ("trainLoss", "iris_trn_rnd", "loss", "train/loss", "celoss", "training cross-entropy"),
                    ("trainAcc", "iris_trn_rnd", "acc", "train/accuracy", "fraction", "training accuracy"),
                    ("objective", "iris_trn_rnd", "obj", "train/objective", "celoss", "the objective Adam minimized — training cross-entropy + wd/2 ‖θ‖² (weight decay 3e-3), so θ* is its minimum —"),
                    ("lossSetosa", "iris_trn_rnd", "loss0", "train/loss/setosa", "celoss", "training cross-entropy of the setosa examples"),
                    ("lossVersicolor", "iris_trn_rnd", "loss1", "train/loss/versicolor", "celoss", "training cross-entropy of the versicolor examples"),
                    ("lossVirginica", "iris_trn_rnd", "loss2", "train/loss/virginica", "celoss", "training cross-entropy of the virginica examples"),
                ]
            },
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
    for p, l, a, t in zip(points2, ref["loss2"], ref["acc2"], ref["train2"]):
        print(f"  2D t={p}: val loss {l:.6f} acc {a:.3f}   train loss {t['loss']:.6f} obj {t['obj']:.6f} per class {[round(c, 4) for c in t['perClass']]}")
    for p, l, a, t in zip(points3, ref["loss3"], ref["acc3"], ref["train3"]):
        print(f"  3D t={p}: val loss {l:.6f} acc {a:.3f}   train loss {t['loss']:.6f} obj {t['obj']:.6f}")


if __name__ == "__main__":
    main()
