"""Compare direct, recursive, and vectorized Strassen under flopscope 0.10.0."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import flopscope as flops
import flopscope.numpy as fnp

LIMIT = 100_000_000_000
LAMBDA = 1e11


def join(c11, c12, c21, c22):
    top = fnp.concatenate((c11, c12), axis=-1)
    bot = fnp.concatenate((c21, c22), axis=-1)
    return fnp.concatenate((top, bot), axis=-2)


def naive(a, b, level):
    m, k, n = int(a.shape[-2]), int(a.shape[-1]), int(b.shape[-1])
    if level <= 0 or (m | k | n) & 1:
        return a @ b
    h, i, j = m // 2, k // 2, n // 2
    a11, a12, a21, a22 = a[:h, :i], a[:h, i:], a[h:, :i], a[h:, i:]
    b11, b12, b21, b22 = b[:i, :j], b[:i, j:], b[i:, :j], b[i:, j:]
    r = level - 1
    p1 = naive(a11 + a22, b11 + b22, r)
    p2 = naive(a21 + a22, b11, r)
    p3 = naive(a11, b12 - b22, r)
    p4 = naive(a22, b21 - b11, r)
    p5 = naive(a11 + a12, b22, r)
    p6 = naive(a21 - a11, b11 + b12, r)
    p7 = naive(a12 - a22, b21 + b22, r)
    return join(p1 + p4 - p5 + p7, p3 + p5, p2 + p4, p1 - p2 + p3 + p6)


def batched(a, b, level):
    m, k, n = int(a.shape[-2]), int(a.shape[-1]), int(b.shape[-1])
    div = 1 << level
    if level <= 0 or m % div or k % div or n % div:
        return a @ b
    aa = fnp.reshape(a, (1, m, k))
    bb = fnp.reshape(b, (1, k, n))
    for _ in range(level):
        p, mm, kk = map(int, aa.shape)
        nn = int(bb.shape[-1])
        h, i, j = mm // 2, kk // 2, nn // 2
        a11, a12, a21, a22 = aa[:, :h, :i], aa[:, :h, i:], aa[:, h:, :i], aa[:, h:, i:]
        b11, b12, b21, b22 = bb[:, :i, :j], bb[:, :i, j:], bb[:, i:, :j], bb[:, i:, j:]
        left = fnp.stack((a11 + a22, a21 + a22, a11, a22, a11 + a12, a21 - a11, a12 - a22), axis=1)
        right = fnp.stack((b11 + b22, b11, b12 - b22, b21 - b11, b22, b11 + b12, b21 + b22), axis=1)
        aa = fnp.reshape(left, (7 * p, h, i))
        bb = fnp.reshape(right, (7 * p, i, j))
    cc = aa @ bb
    for _ in range(level):
        p, h, j = int(cc.shape[0]) // 7, int(cc.shape[-2]), int(cc.shape[-1])
        q = fnp.reshape(cc, (p, 7, h, j))
        p1, p2, p3, p4, p5, p6, p7 = (q[:, x] for x in range(7))
        cc = join(p1 + p4 - p5 + p7, p3 + p5, p2 + p4, p1 - p2 + p3 + p6)
    return cc[0]


def run(name, fn, a, b, ref, repeats):
    rows = []
    for rep in range(repeats):
        flops.budget_reset()
        with flops.BudgetContext(LIMIT, quiet=True, namespace=name) as ctx:
            out = fn(a, b)
        got = np.asarray(out)
        delta = got.astype(np.float64) - ref.astype(np.float64)
        summary = ctx.summary_dict()
        residual = float(summary["residual_wall_time_s"] or 0.0)
        used = int(summary["flops_used"])
        rows.append({
            "repeat": rep,
            "flops": used,
            "residual_s": residual,
            "effective_compute": used + LAMBDA * residual,
            "backend_s": float(summary["flopscope_backend_time_s"]),
            "overhead_s": float(summary["flopscope_overhead_time_s"]),
            "wall_s": float(summary["wall_time_s"] or 0.0),
            "relative_rmse": float(np.sqrt(np.mean(delta * delta)) / np.sqrt(np.mean(ref.astype(np.float64) ** 2))),
            "max_abs_error": float(np.max(np.abs(delta))),
        })
        del out, got, delta
        gc.collect()
    return {"name": name, "best": min(rows, key=lambda x: x["effective_compute"]), "attempts": rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=16384)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--output", type=Path, default=Path("research/results/strassen_meter.json"))
    args = p.parse_args()
    rng = np.random.default_rng(20260810)
    a0 = rng.standard_normal((256, 256), dtype=np.float32)
    b0 = rng.standard_normal((256, args.samples), dtype=np.float32)
    ref = a0 @ b0
    flops.budget_reset()
    with flops.BudgetContext(LIMIT, quiet=True):
        a, b = fnp.asarray(a0), fnp.asarray(b0)
    tests = [
        ("direct", lambda x, y: x @ y),
        ("naive_l2", lambda x, y: naive(x, y, 2)),
        ("naive_l3", lambda x, y: naive(x, y, 3)),
        ("batched_l2", lambda x, y: batched(x, y, 2)),
        ("batched_l3", lambda x, y: batched(x, y, 3)),
        ("batched_l4", lambda x, y: batched(x, y, 4)),
    ]
    records = []
    for name, fn in tests:
        print("START", name, flush=True)
        try:
            record = run(name, fn, a, b, ref, args.repeats)
        except Exception as exc:
            record = {"name": name, "error_type": type(exc).__name__, "error": str(exc)}
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    base = next(r["best"]["effective_compute"] for r in records if r["name"] == "direct")
    for r in records:
        if "best" in r:
            r["ratio_to_direct"] = r["best"]["effective_compute"] / base
            r["speedup_vs_direct"] = base / r["best"]["effective_compute"]
    payload = {"samples": args.samples, "repeats": args.repeats, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
