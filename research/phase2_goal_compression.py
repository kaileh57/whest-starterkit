#!/usr/bin/env python3
"""Ablate goal-directed signed recompression of factored K=3 state.

This is a research harness, not a submission estimator. It vendors Paul Rosu's
public flopscope-native K=3 implementation at runtime, then compares:

  * uncapped factored K=3,
  * ordinary magnitude/top-J truncation,
  * deterministic signed recompression chosen from downstream observables.

The primary metric is fidelity to uncapped K=3 on unseen generated MLPs. A
radially integrated, antithetic scrambled-Sobol estimate is used as a secondary
truth check. No benchmark labels or challenge targets enter the compressor.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import qmc

PAUL_RAW = (
    "https://raw.githubusercontent.com/paulrosu11/arc-cumulant-mlp-estimator/"
    "54ec39e0bb1228556016fc70f76e9edc10f8a82e/"
    "whitebox_estimator/algorithm/estimator.py"
)


def load_reference(cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "paul_k3_estimator.py"
    if not path.exists():
        print(f"downloading {PAUL_RAW}", flush=True)
        urllib.request.urlretrieve(PAUL_RAW, path)
    spec = importlib.util.spec_from_file_location("paul_k3_estimator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def he_weights(width: int, depth: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    std = math.sqrt(2.0 / width)
    return [rng.normal(0.0, std, (width, width)).astype(np.float32) for _ in range(depth)]


def chi_mean(dim: int) -> float:
    return math.sqrt(2.0) * math.exp(math.lgamma((dim + 1.0) / 2.0) - math.lgamma(dim / 2.0))


def spherical_qmc_mean(
    weights: list[np.ndarray], *, exponent: int, seed: int, batch: int = 4096
) -> np.ndarray:
    """Estimate all layer means using exact radial integration.

    A zero-bias ReLU MLP is positively homogeneous, so for X=R U with U
    uniform on the sphere, E[f(X)] = E[R] E[f(U)]. We sample only U and use
    explicit antipodes. `exponent` is the total sample exponent: 2**exponent
    directions are evaluated after antipodal doubling.
    """
    width = weights[0].shape[0]
    if exponent < 2:
        raise ValueError("exponent must be >= 2")
    sobol = qmc.Sobol(width, scramble=True, seed=seed)
    z = ndtri(np.clip(sobol.random_base2(exponent - 1), 1e-12, 1.0 - 1e-12))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    directions = np.concatenate((z, -z), axis=0)
    rows = np.zeros((len(weights), width), dtype=np.float64)
    for start in range(0, len(directions), batch):
        x = directions[start : start + batch]
        for layer, W in enumerate(weights):
            x = np.maximum(x @ W.astype(np.float64), 0.0)
            rows[layer] += x.sum(axis=0)
    rows *= chi_mean(width) / len(directions)
    return rows


def k2_trajectory(weights: list[np.ndarray]) -> dict[str, Any]:
    """Official covariance-style Gaussian closure plus adjoint coefficients."""
    width = weights[0].shape[0]
    mu = np.zeros(width, dtype=np.float64)
    cov = np.eye(width, dtype=np.float64)
    rows: list[np.ndarray] = []
    gains: list[np.ndarray] = []
    h3: list[np.ndarray] = []
    pre_means: list[np.ndarray] = []
    pre_vars: list[np.ndarray] = []

    for W32 in weights:
        W = W32.astype(np.float64)
        pre_mu = W.T @ mu
        pre_cov = W.T @ cov @ W
        var = np.maximum(np.diag(pre_cov), 1e-12)
        sigma = np.sqrt(var)
        alpha = pre_mu / sigma
        phi = np.exp(-0.5 * alpha * alpha) / math.sqrt(2.0 * math.pi)
        Phi = ndtr(alpha)
        out_mu = pre_mu * Phi + sigma * phi
        second = (pre_mu * pre_mu + var) * Phi + pre_mu * sigma * phi
        out_var = np.maximum(second - out_mu * out_mu, 0.0)
        out_cov = np.outer(Phi, Phi) * pre_cov
        np.fill_diagonal(out_cov, out_var)
        out_cov = 0.5 * (out_cov + out_cov.T)

        pre_means.append(pre_mu)
        pre_vars.append(var)
        gains.append(Phi)
        # E[d^3/dz^3 ReLU(z)] under N(pre_mu,var).
        h3.append(-alpha * phi / var)
        rows.append(out_mu)
        mu, cov = out_mu, out_cov

    depth = len(weights)
    q_to_final_pre: list[np.ndarray | None] = [None] * depth
    if depth >= 2:
        Q = weights[-1].astype(np.float64).T
        q_to_final_pre[depth - 2] = Q.copy()
        for layer in range(depth - 3, -1, -1):
            local = gains[layer + 1][:, None] * weights[layer + 1].astype(np.float64).T
            Q = Q @ local
            q_to_final_pre[layer] = Q.copy()

    return {
        "rows": np.stack(rows),
        "gains": gains,
        "h3": h3,
        "q_to_final_pre": q_to_final_pre,
        "pre_means": pre_means,
        "pre_vars": pre_vars,
    }


def normalized_block(block: np.ndarray) -> np.ndarray:
    """Normalize a feature block by the norm of its aggregate signature."""
    total = block.sum(axis=1)
    scale = np.linalg.norm(total)
    if not np.isfinite(scale) or scale < 1e-20:
        scale = np.linalg.norm(block)
    return block / max(float(scale), 1e-20)


def signed_omp(features: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Select columns and signed weights to preserve their aggregate.

    Greedy selection is deterministic. At each step, the coefficients are
    refit by a tiny ridge solve. The target is exactly the sum of all feature
    columns, so no labels are involved.
    """
    _, count = features.shape
    if rank >= count:
        return np.arange(count), np.ones(count), 0.0
    target = features.sum(axis=1)
    target_norm = max(float(np.linalg.norm(target)), 1e-20)
    col_norm = np.linalg.norm(features, axis=0)
    selected: list[int] = []
    available = np.ones(count, dtype=bool)
    residual = target.copy()
    weights = np.empty(0, dtype=np.float64)

    for _ in range(rank):
        score = np.abs(features.T @ residual) / np.maximum(col_norm, 1e-20)
        score[~available] = -np.inf
        idx = int(np.argmax(score))
        if not np.isfinite(score[idx]):
            break
        selected.append(idx)
        available[idx] = False
        X = features[:, selected]
        gram = X.T @ X
        ridge = 1e-10 * max(float(np.trace(gram)) / max(len(selected), 1), 1e-20)
        weights = np.linalg.solve(gram + ridge * np.eye(len(selected)), X.T @ target)
        residual = target - X @ weights
        if np.linalg.norm(residual) / target_norm < 1e-8:
            break

    return np.asarray(selected, dtype=np.int64), weights, float(np.linalg.norm(residual) / target_norm)


def goal_compress(module, K3, Q: np.ndarray, h3_final: np.ndarray, rank: int):
    """Compress CP columns against local and final target-side observables."""
    A_f, B_f, C_f = K3.factors
    A = np.asarray(A_f, dtype=np.float64)
    B = np.asarray(B_f, dtype=np.float64)
    C = np.asarray(C_f, dtype=np.float64)
    count = A.shape[1]
    if count <= rank:
        return K3, {"before": count, "after": count, "residual": 0.0, "weight_l1": float(count)}

    # Final direct K3 readout under the K2 linearized transport.
    QA, QB, QC = Q @ A, Q @ B, Q @ C
    final_signature = h3_final[:, None] * QA * QB * QC

    # Preserve the current repeated diagonal as a local trajectory guard.
    local_diag = A * B * C

    features = np.concatenate(
        (normalized_block(final_signature), normalized_block(local_diag)), axis=0
    )
    idx, signed_weight, residual = signed_omp(features, rank)
    if idx.size == 0:
        raise RuntimeError("goal compressor selected no columns")

    # A CP term is multilinear, so absorb each signed coefficient into one leg.
    A_new = A[:, idx] * signed_weight[None, :]
    B_new = B[:, idx]
    C_new = C[:, idx]
    compressed = module.FactoredTensor3(
        n=K3.n,
        factors=(
            module.fnp.asarray(A_new, dtype=module.fnp.float32),
            module.fnp.asarray(B_new, dtype=module.fnp.float32),
            module.fnp.asarray(C_new, dtype=module.fnp.float32),
        ),
    )
    return compressed, {
        "before": count,
        "after": int(idx.size),
        "residual": residual,
        "weight_l1": float(np.abs(signed_weight).sum()),
        "weight_max": float(np.abs(signed_weight).max(initial=0.0)),
    }


def predict_goal_compressed(module, weights: list[np.ndarray], rank: int):
    trajectory = k2_trajectory(weights)
    q_list = trajectory["q_to_final_pre"]
    h3_final = trajectory["h3"][-1]
    width = weights[0].shape[0]
    depth = len(weights)
    K = {
        1: module.fnp.zeros(width, dtype=module.fnp.float32),
        2: module.fnp.eye(width, dtype=module.fnp.float32),
    }
    rows = []
    diagnostics = []
    for layer, W in enumerate(weights):
        is_last = layer == depth - 1
        K_pre = module._linear_step(K, module.fnp.asarray(W, dtype=module.fnp.float32))
        K = module._factored_nonlin_simple_drops(
            K_pre, last_layer_lite=is_last, drops=True
        )
        rows.append(np.asarray(K[1], dtype=np.float64))
        if not is_last and 3 in K:
            Q = q_list[layer]
            if Q is None:
                raise RuntimeError(f"missing adjoint map for layer {layer}")
            K[3], info = goal_compress(module, K[3], Q, h3_final, rank)
            info["layer"] = layer
            diagnostics.append(info)
    return np.stack(rows), diagnostics


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) ** 2))


def geometric_mean(values: list[float]) -> float:
    positive = np.maximum(np.asarray(values, dtype=np.float64), 1e-300)
    return float(np.exp(np.mean(np.log(positive))))


def run_one(module, width: int, depth: int, seed: int, ranks: list[int], qmc_exp: int):
    print(f"width={width} depth={depth} seed={seed}", flush=True)
    weights = he_weights(width, depth, seed)

    t0 = time.perf_counter()
    truth_a = spherical_qmc_mean(weights, exponent=qmc_exp, seed=100_000 + seed)
    truth_b = spherical_qmc_mean(weights, exponent=qmc_exp, seed=200_000 + seed)
    truth = 0.5 * (truth_a + truth_b)
    qmc_seconds = time.perf_counter() - t0
    qmc_noise = mse(truth_a[-1], truth_b[-1]) / 4.0

    t0 = time.perf_counter()
    k2 = k2_trajectory(weights)["rows"]
    k2_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    full = np.asarray(
        module.predict_k3_factored_simple_drops(
            weights, drops=True, lite_last=True, J_max=None, eps_factor=0.0
        ),
        dtype=np.float64,
    )
    full_seconds = time.perf_counter() - t0

    variants: dict[str, Any] = {}
    for rank in ranks:
        t0 = time.perf_counter()
        magnitude = np.asarray(
            module.predict_k3_factored_simple_drops(
                weights, drops=True, lite_last=True, J_max=rank, eps_factor=0.0
            ),
            dtype=np.float64,
        )
        magnitude_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        goal, diagnostics = predict_goal_compressed(module, weights, rank)
        goal_seconds = time.perf_counter() - t0

        variants[str(rank)] = {
            "magnitude_fidelity_mse": mse(magnitude[-1], full[-1]),
            "goal_fidelity_mse": mse(goal[-1], full[-1]),
            "magnitude_truth_mse": mse(magnitude[-1], truth[-1]),
            "goal_truth_mse": mse(goal[-1], truth[-1]),
            "magnitude_seconds": magnitude_seconds,
            "goal_seconds": goal_seconds,
            "goal_diagnostics": diagnostics,
        }
        print(
            f"  rank={rank:4d} fidelity magnitude={variants[str(rank)]['magnitude_fidelity_mse']:.3e} "
            f"goal={variants[str(rank)]['goal_fidelity_mse']:.3e}",
            flush=True,
        )

    return {
        "width": width,
        "depth": depth,
        "seed": seed,
        "qmc_exponent_per_scramble": qmc_exp,
        "qmc_noise_mse_estimate": qmc_noise,
        "k2_truth_mse": mse(k2[-1], truth[-1]),
        "full_k3_truth_mse": mse(full[-1], truth[-1]),
        "k2_seconds": k2_seconds,
        "full_k3_seconds": full_seconds,
        "qmc_seconds": qmc_seconds,
        "variants": variants,
    }


def aggregate(cases: list[dict[str, Any]], ranks: list[int]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "cases": len(cases),
        "k2_truth_mse_geomean": geometric_mean([c["k2_truth_mse"] for c in cases]),
        "full_k3_truth_mse_geomean": geometric_mean([c["full_k3_truth_mse"] for c in cases]),
        "qmc_noise_mse_geomean": geometric_mean([c["qmc_noise_mse_estimate"] for c in cases]),
        "ranks": {},
    }
    for rank in ranks:
        key = str(rank)
        mag_fid = [c["variants"][key]["magnitude_fidelity_mse"] for c in cases]
        goal_fid = [c["variants"][key]["goal_fidelity_mse"] for c in cases]
        mag_truth = [c["variants"][key]["magnitude_truth_mse"] for c in cases]
        goal_truth = [c["variants"][key]["goal_truth_mse"] for c in cases]
        ratios = [g / max(m, 1e-300) for g, m in zip(goal_fid, mag_fid)]
        result["ranks"][key] = {
            "magnitude_fidelity_geomean": geometric_mean(mag_fid),
            "goal_fidelity_geomean": geometric_mean(goal_fid),
            "goal_over_magnitude_fidelity_geomean": geometric_mean(ratios),
            "goal_fidelity_wins": int(sum(g < m for g, m in zip(goal_fid, mag_fid))),
            "magnitude_truth_geomean": geometric_mean(mag_truth),
            "goal_truth_geomean": geometric_mean(goal_truth),
            "goal_truth_wins": int(sum(g < m for g, m in zip(goal_truth, mag_truth))),
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=int(os.environ.get("WIDTH", "32")))
    parser.add_argument("--depth", type=int, default=int(os.environ.get("DEPTH", "16")))
    parser.add_argument("--seeds", type=int, default=int(os.environ.get("SEEDS", "6")))
    parser.add_argument("--seed-offset", type=int, default=int(os.environ.get("SEED_OFFSET", "7000")))
    parser.add_argument("--ranks", default=os.environ.get("RANKS", "8,16,32"))
    parser.add_argument("--qmc-exp", type=int, default=int(os.environ.get("QMC_EXP", "14")))
    parser.add_argument("--output", default="research/results/phase2_goal_compression.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ranks = sorted({int(x) for x in args.ranks.split(",") if x.strip()})
    module = load_reference(Path("research/vendor"))
    cases = []
    for i in range(args.seeds):
        cases.append(
            run_one(
                module,
                width=args.width,
                depth=args.depth,
                seed=args.seed_offset + i,
                ranks=ranks,
                qmc_exp=args.qmc_exp,
            )
        )
    report = {
        "method": "goal-directed signed CP recompression",
        "reference_commit": "54ec39e0bb1228556016fc70f76e9edc10f8a82e",
        "config": vars(args),
        "aggregate": aggregate(cases, ranks),
        "cases": cases,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["aggregate"], indent=2), flush=True)
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
