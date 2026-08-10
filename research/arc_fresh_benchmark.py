#!/usr/bin/env python3
"""Benchmark ARC cumulant propagation on fresh, non-public 256x32 networks.

The target is estimated with independent Haar-randomized full Kerdock/MUB
spherical 5-designs.  Every randomized design is unbiased over its Haar
rotation, so for a deterministic analytic estimate ``a`` we use

    mean_j [(a_j - mean_r P_rj)^2 - Var_r(P_rj)/R]

which is unbiased for the per-neuron squared error, conditional on the network.
No public WhestBench targets or network seeds are used.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from mub_design import build_unit_bases

WIDTH = 256
DEPTH = 32
WEIGHT_SCALE = np.float32(math.sqrt(2.0 / WIDTH))
RADIUS_MEAN = np.float32(
    math.exp(
        0.5 * math.log(2.0)
        + math.lgamma((WIDTH + 1.0) / 2.0)
        - math.lgamma(WIDTH / 2.0)
    )
)


def haar_q(rng: np.random.Generator) -> np.ndarray:
    raw = rng.standard_normal((WIDTH, WIDTH))
    q, r = np.linalg.qr(raw)
    signs = np.where(np.diag(r) >= 0.0, 1.0, -1.0)
    return np.asarray(q * signs[None, :], dtype=np.float32)


def fresh_weights(seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [
        rng.standard_normal((WIDTH, WIDTH), dtype=np.float32) * WEIGHT_SCALE
        for _ in range(DEPTH)
    ]


def forward_mean(half_np: np.ndarray, weights: list[np.ndarray]) -> np.ndarray:
    half = torch.from_numpy(half_np)
    ws = [torch.from_numpy(w) for w in weights]
    with torch.inference_mode():
        z = half @ ws[0]
        a = torch.relu(z)
        y = torch.cat((a, a - z), dim=0)
        for w in ws[1:]:
            y = torch.relu(y @ w)
        return y.mean(dim=0).cpu().numpy().astype(np.float64)


def randomized_design_estimates(
    bases: np.ndarray,
    weights: list[np.ndarray],
    *,
    rotations: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    flat = bases.reshape(-1, WIDTH)
    predictions = []
    for rep in range(rotations):
        q = haar_q(rng)
        half = np.ascontiguousarray((flat @ q) * RADIUS_MEAN, dtype=np.float32)
        t0 = time.perf_counter()
        predictions.append(forward_mean(half, weights))
        print(
            f"reference rep={rep + 1}/{rotations} seconds={time.perf_counter() - t0:.2f}",
            flush=True,
        )
    return np.stack(predictions, axis=0)


def corrected_mse(estimate: np.ndarray, references: np.ndarray) -> dict[str, float]:
    count = references.shape[0]
    ref_mean = references.mean(axis=0)
    ref_var = references.var(axis=0, ddof=1)
    raw = (estimate - ref_mean) ** 2
    corrected_by_neuron = raw - ref_var / float(count)
    return {
        "corrected_raw_mse": float(corrected_by_neuron.mean()),
        "uncorrected_vs_reference_mean_mse": float(raw.mean()),
        "mean_reference_variance_single_design": float(ref_var.mean()),
        "mean_reference_variance_of_mean": float(ref_var.mean() / count),
        "median_corrected_neuron_sqerr": float(np.median(corrected_by_neuron)),
        "fraction_negative_corrected_neurons": float(np.mean(corrected_by_neuron < 0.0)),
    }


def build_arc_mlp(weights: list[np.ndarray], dtype: torch.dtype):
    from mlp_kprop.mlp import MLP

    # ARC's MLP omits the nonlinearity after its final linear layer.  Add an
    # identity 33rd linear layer so act31 of the challenge is pre32 in ARC's
    # convention.
    mlp = MLP(
        input_dim=WIDTH,
        hidden_dim=WIDTH,
        output_dim=WIDTH,
        num_layers=DEPTH + 1,
        nonlin="relu",
        init_kind="manual",
        w_var=[2.0] * DEPTH + [1.0],
        b_var=0.0,
        b_mean=0.0,
    ).to(dtype=dtype)
    with torch.no_grad():
        for layer, w in enumerate(weights):
            # Challenge convention is x @ W; torch Linear stores W^T.
            mlp.Ws[layer].weight.copy_(torch.from_numpy(w.T).to(dtype=dtype))
        mlp.Ws[-1].weight.copy_(torch.eye(WIDTH, dtype=dtype))
    return mlp


def run_arc_variant(mlp, variant: str) -> tuple[np.ndarray, float]:
    from mlp_kprop.harmonic import HTensor
    from mlp_kprop.kprop_harmonic import AUGMENT, SIMPLE, mlp_kprop

    if variant == "k2_simple":
        k_max, kind, factor = 2, SIMPLE, False
    elif variant == "k3_simple_factor":
        k_max, kind, factor = 3, SIMPLE, True
    elif variant == "k3_augment_factor":
        k_max, kind, factor = 3, AUGMENT, True
    elif variant == "k4_simple_factor":
        k_max, kind, factor = 4, SIMPLE, True
    elif variant == "k4_augment_factor":
        k_max, kind, factor = 4, AUGMENT, True
    else:
        raise ValueError(variant)

    dtype = next(mlp.parameters()).dtype
    K_in = {
        1: torch.zeros(WIDTH, dtype=dtype),
        2: torch.eye(WIDTH, dtype=dtype),
    }
    t0 = time.perf_counter()
    with torch.inference_mode():
        result = mlp_kprop(
            mlp,
            K_in,
            k_max=k_max,
            kind=kind,
            factor=factor,
            use_avg_metric=False,
            output_d_max=1,
        )
        mean_obj = result[1]
        if isinstance(mean_obj, HTensor):
            mean = mean_obj.to_tensor()
        else:
            mean = mean_obj
    elapsed = time.perf_counter() - t0
    return mean.detach().cpu().numpy().astype(np.float64), elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arc-src", type=Path, required=True)
    parser.add_argument("--networks", type=int, default=2)
    parser.add_argument("--rotations", type=int, default=6)
    parser.add_argument("--seed-base", type=int, default=95_000_000)
    parser.add_argument(
        "--variants",
        default="k2_simple,k3_simple_factor,k3_augment_factor",
    )
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument(
        "--output", type=Path, default=Path("research/results/arc_fresh_benchmark.json")
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.arc_src.resolve()))
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    bases = build_unit_bases()
    records: list[dict[str, object]] = []
    start = time.perf_counter()

    for network_index in range(args.networks):
        network_seed = args.seed_base + network_index
        print(f"NETWORK {network_index + 1}/{args.networks} seed={network_seed}", flush=True)
        weights = fresh_weights(network_seed)
        references = randomized_design_estimates(
            bases,
            weights,
            rotations=args.rotations,
            seed=network_seed ^ 0x5245464552454E43,
        )
        reference_var = float(references.var(axis=0, ddof=1).mean())
        mlp = build_arc_mlp(weights, dtype=torch.float64)
        methods: dict[str, object] = {}
        for variant in variants:
            print(f"ARC_START variant={variant}", flush=True)
            try:
                estimate, elapsed = run_arc_variant(mlp, variant)
                metrics = corrected_mse(estimate, references)
                metrics["seconds"] = elapsed
                metrics["finite"] = bool(np.isfinite(estimate).all())
                metrics["mean_prediction"] = float(estimate.mean())
                methods[variant] = metrics
                print(
                    f"ARC_RESULT variant={variant} seconds={elapsed:.2f} "
                    f"mse={metrics['corrected_raw_mse']:.9e}",
                    flush=True,
                )
            except Exception as exc:
                methods[variant] = {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                print(
                    f"ARC_ERROR variant={variant} type={type(exc).__name__} error={exc}",
                    flush=True,
                )
        record = {
            "network_index": network_index,
            "network_seed": network_seed,
            "reference_single_design_mse": reference_var,
            "methods": methods,
        }
        records.append(record)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "protocol": {
                        "width": WIDTH,
                        "depth": DEPTH,
                        "network_seed_base": args.seed_base,
                        "network_count": args.networks,
                        "reference_rotations": args.rotations,
                        "public_dataset_used": False,
                        "selection_role": "dev_baseline_recovery",
                        "arc_src": str(args.arc_src),
                        "variants": variants,
                    },
                    "records": records,
                    "elapsed_seconds": time.perf_counter() - start,
                },
                indent=2,
                sort_keys=True,
            )
        )

    aggregates: dict[str, object] = {}
    for variant in variants:
        values = []
        seconds = []
        for record in records:
            method = record["methods"].get(variant, {})
            if "corrected_raw_mse" in method:
                values.append(float(method["corrected_raw_mse"]))
                seconds.append(float(method["seconds"]))
        if values:
            aggregates[variant] = {
                "mean_corrected_raw_mse": float(np.mean(values)),
                "median_corrected_raw_mse": float(np.median(values)),
                "mean_seconds": float(np.mean(seconds)),
                "successful_networks": len(values),
            }

    payload = json.loads(args.output.read_text())
    payload["aggregates"] = aggregates
    payload["elapsed_seconds"] = time.perf_counter() - start
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({"aggregates": aggregates}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
