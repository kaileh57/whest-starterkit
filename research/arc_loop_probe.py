#!/usr/bin/env python3
"""Identify the exact covariance-triangle term omitted by factorized augmented K3.

This is a source-level scientific probe against ARC's current cumulant-propagation
implementation.  We deliberately begin from a Gaussian preactivation state: K1 and
K2 may be nontrivial, while every cumulant of degree >=3 is zero.  In that regime,
the only non-hypertree augmented diagram contributing to the output all-distinct
third cumulant is the covariance triangle.  Therefore

    full_unfactorized_AUGMENT - factorized_AUGMENT

isolates that term without fitting or target leakage.

The script checks the candidate closed form

    Delta K3[i,j,k] = c * q2[i] q2[j] q2[k] C[i,j] C[j,k] C[k,i]

on all-distinct indices, recovers c, and reports machine-scale residuals.  It also
prints ARC's exact dropped diagram list so any additional term is explicit.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch


def _all_distinct_mask(n: int, *, device: torch.device) -> torch.Tensor:
    ids = torch.arange(n, device=device)
    i = ids[:, None, None]
    j = ids[None, :, None]
    k = ids[None, None, :]
    return (i != j) & (i != k) & (j != k)


def _tensor_metrics(target: torch.Tensor, candidate: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    t = target[mask]
    x = candidate[mask]
    xx = torch.dot(x, x)
    alpha = torch.dot(t, x) / xx if float(xx) > 0.0 else torch.tensor(float("nan"), dtype=t.dtype)
    residual = t - alpha * x
    denom = torch.dot(t, t)
    return {
        "coefficient": float(alpha),
        "relative_squared_residual": float(torch.dot(residual, residual) / denom),
        "max_abs_residual": float(torch.max(torch.abs(residual))),
        "target_rms": float(torch.sqrt(torch.mean(t * t))),
        "candidate_rms": float(torch.sqrt(torch.mean(x * x))),
        "correlation": float(torch.corrcoef(torch.stack((t, x)))[0, 1]),
    }


def _dropped_terms(kh: Any) -> list[dict[str, Any]]:
    terms = kh.get_all_terms_iso(k_max=3, d_max=4, augment=True)
    dropped: list[dict[str, Any]] = []
    for int_part, classes in terms.items():
        for vec_part, count in classes.items():
            if not kh.factored_keeps_term(3, int_part, vec_part):
                dropped.append(
                    {
                        "int_part": tuple(int(x) for x in int_part),
                        "vec_part": tuple(tuple(int(y) for y in x) for x in vec_part),
                        "count": int(count),
                        "hypertree": bool(kh.is_hypertree(vec_part)),
                        "block_cost": int(sum(kh.vec_block_cost(v) for v in vec_part)),
                    }
                )
    return dropped


def one_case(seed: int, n: int, use_avg_metric: bool) -> dict[str, Any]:
    from mlp_kprop.factor_k3 import factored_nonlin_kprop_k3
    from mlp_kprop.harmonic import HTensor
    from mlp_kprop.kprop_harmonic import Kind, coerce_input, linear_kprop, nonlin_kprop
    from mlp_kprop.wick import relu_wick_coef

    torch.manual_seed(seed)
    dtype = torch.float64
    device = torch.device("cpu")

    # Standard Gaussian input followed by a generic linear map gives an exactly
    # Gaussian preactivation with dense covariance, while all higher cumulants vanish.
    K0 = coerce_input(
        {1: torch.zeros(n, dtype=dtype, device=device), 2: torch.eye(n, dtype=dtype, device=device)},
        k_max=3,
        kind=Kind.AUGMENT,
    )
    W = torch.randn(n, n, dtype=dtype, device=device) * math.sqrt(2.0 / n)
    metric = 2.0 * torch.ones(n, dtype=dtype, device=device) if use_avg_metric else None
    WK = linear_kprop(K0, W, k_max=3, set_metric=metric)

    full = nonlin_kprop(
        WK,
        nonlin_wick_coef=relu_wick_coef,
        k_max=3,
        kind=Kind.AUGMENT,
    )
    factored = factored_nonlin_kprop_k3(
        K_in=WK,
        nonlin_wick_coef=relu_wick_coef,
        augment=True,
    )

    full3 = full[3].to_tensor()
    fact3 = factored[3].to_tensor()
    delta = full3 - fact3

    mean = WK[1].core
    C = WK[2].core
    if isinstance(C, HTensor):
        C = C.to_tensor()
    var = torch.diagonal(C)
    q2 = relu_wick_coef(mean=mean, var=var, k=2, p=1)

    candidate = (
        q2[:, None, None]
        * q2[None, :, None]
        * q2[None, None, :]
        * C[:, :, None]
        * C[None, :, :]
        * C.T[:, None, :]
    )
    mask = _all_distinct_mask(n, device=device)
    metrics = _tensor_metrics(delta, candidate, mask)

    # Repeated-index slices are intentionally reported separately: pK-to-K conversion
    # and zero-repeated conventions can add Möbius corrections there.
    ids = torch.arange(n)
    d3 = delta[ids, ids, ids]
    d21 = delta[ids[:, None], ids[:, None], ids[None, :]]
    repeated = {
        "diag_rms": float(torch.sqrt(torch.mean(d3 * d3))),
        "two_one_rms": float(torch.sqrt(torch.mean(d21 * d21))),
        "full_delta_rms": float(torch.sqrt(torch.mean(delta * delta))),
    }

    return {
        "seed": seed,
        "n": n,
        "use_avg_metric": use_avg_metric,
        "metrics_all_distinct": metrics,
        "repeated_slices": repeated,
        "mean_rms": float(torch.sqrt(torch.mean(mean * mean))),
        "covariance_rms": float(torch.sqrt(torch.mean(C * C))),
        "q2_rms": float(torch.sqrt(torch.mean(q2 * q2))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arc-src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sizes", default="6,8,12,16")
    parser.add_argument("--seeds", default="101,202,303")
    args = parser.parse_args()

    sys.path.insert(0, str(args.arc_src))
    torch.set_default_dtype(torch.float64)
    torch.set_grad_enabled(False)

    import mlp_kprop.kprop_harmonic as kh

    results: dict[str, Any] = {
        "arc_source": str(args.arc_src),
        "dropped_terms": _dropped_terms(kh),
        "cases": [],
    }
    for n in (int(x) for x in args.sizes.split(",") if x):
        for seed in (int(x) for x in args.seeds.split(",") if x):
            for avg in (False, True):
                case = one_case(seed=seed, n=n, use_avg_metric=avg)
                results["cases"].append(case)
                m = case["metrics_all_distinct"]
                print(
                    f"n={n:2d} seed={seed:3d} avg={avg} "
                    f"coef={m['coefficient']:.12g} residual={m['relative_squared_residual']:.3e} "
                    f"corr={m['correlation']:.12f}",
                    flush=True,
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
