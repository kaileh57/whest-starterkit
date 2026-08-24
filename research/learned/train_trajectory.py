from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from torch import Tensor, nn


DATASET_ID = "aicrowd/arc-whestbench-public-2026"
DATASET_REVISION = "v1-phase1"
SQRT_2 = math.sqrt(2.0)
SQRT_2PI = math.sqrt(2.0 * math.pi)


@dataclass
class Config:
    dataset: str
    train_count: int
    val_count: int
    hidden: int
    epochs: int
    seed: int
    output: str
    learning_rate: float = 1.5e-3
    weight_decay: float = 2.0e-5
    grad_accum: int = 4
    final_weight: float = 32.0
    correction_scale: float = 0.20


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))


def gaussian_relu(mu: Tensor, var: Tensor) -> tuple[Tensor, Tensor]:
    var = var.clamp_min(1.0e-12)
    sigma = var.sqrt()
    a = mu / sigma
    phi = torch.exp(-0.5 * a.square()) / SQRT_2PI
    Phi = 0.5 * (1.0 + torch.erf(a / SQRT_2))
    mean = mu * Phi + sigma * phi
    second = (mu.square() + var) * Phi + mu * sigma * phi
    post_var = (second - mean.square()).clamp_min(1.0e-12)
    return mean, post_var


class TrajectoryClosure(nn.Module):
    """Permutation-equivariant learned memory closure around diagonal K2.

    Hidden channels live on the current layer's neurons. They move to the next
    layer only through W and W^2 messages, so independent neuron permutations
    at every hidden layer act equivariantly. The base state is ordinary
    diagonal Gaussian moment propagation; learned corrections are bounded and
    multiplicative, making the zero-correction initialization exactly stable.
    """

    def __init__(self, hidden: int, depth: int = 32, correction_scale: float = 0.20) -> None:
        super().__init__()
        self.hidden = hidden
        self.depth = depth
        self.correction_scale = correction_scale
        # a, log(v_pre), normalized base mean/variance, four column statistics,
        # depth coordinate, W-message, W^2-message.
        feature_dim = 9 + 2 * hidden
        trunk = max(40, 3 * hidden)
        self.local = nn.Sequential(
            nn.Linear(feature_dim, trunk),
            nn.SiLU(),
            nn.Linear(trunk, trunk),
            nn.SiLU(),
            nn.Linear(trunk, hidden + 2),
        )
        last = self.local[-1]
        assert isinstance(last, nn.Linear)
        with torch.no_grad():
            nn.init.normal_(last.weight[:hidden], mean=0.0, std=0.015)
            last.bias[:hidden].zero_()
            last.weight[hidden:].zero_()
            last.bias[hidden:].zero_()

    def forward(self, weights: Tensor, *, return_base: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        # weights: (depth, input_neuron, output_neuron)
        depth, width, width2 = weights.shape
        if width != width2:
            raise ValueError(f"expected square weights, got {tuple(weights.shape)}")
        m = torch.zeros(width, dtype=weights.dtype, device=weights.device)
        v = torch.ones(width, dtype=weights.dtype, device=weights.device)
        h = torch.zeros((width, self.hidden), dtype=weights.dtype, device=weights.device)
        rows: list[Tensor] = []
        base_rows: list[Tensor] = []

        for layer, w in enumerate(weights):
            w2 = w.square()
            mu = m @ w
            v_pre = v @ w2
            base_m, base_v = gaussian_relu(mu, v_pre)
            sigma = v_pre.clamp_min(1.0e-12).sqrt()
            a = mu / sigma

            # Signed message preserves odd/path information. The W^2 message
            # is a normalized positive aggregation suited to moment memory.
            msg_signed = (w.T @ h) / SQRT_2
            col_energy = w2.sum(dim=0).clamp_min(1.0e-8)
            msg_square = (w2.T @ h) / col_energy[:, None]

            col_sum = w.sum(dim=0) / math.sqrt(width * 2.0 / width)
            col_skew = w.pow(3).sum(dim=0) / col_energy.pow(1.5)
            col_kurt = w2.square().sum(dim=0) / col_energy.square()
            col_abs = w.abs().sum(dim=0) / math.sqrt(width)
            depth_coord = torch.full_like(a, (layer + 1.0) / depth)

            local = torch.stack(
                (
                    a,
                    torch.log(v_pre.clamp_min(1.0e-12)),
                    base_m / sigma,
                    base_v / v_pre.clamp_min(1.0e-12),
                    col_sum,
                    col_energy - 2.0,
                    col_skew,
                    col_kurt,
                    depth_coord,
                ),
                dim=1,
            )
            features = torch.cat((local, msg_signed, msg_square), dim=1)
            raw = self.local(features)
            h = torch.tanh(raw[:, : self.hidden])
            dm = raw[:, self.hidden]
            dv = raw[:, self.hidden + 1]

            # Multiplicative corrections preserve positivity and cap any one
            # layer's intervention while still allowing coherent depth effects.
            m = base_m * torch.exp(self.correction_scale * torch.tanh(dm))
            v = base_v * torch.exp(self.correction_scale * torch.tanh(dv))
            rows.append(m)
            base_rows.append(base_m)

        pred = torch.stack(rows)
        if return_base:
            return pred, torch.stack(base_rows)
        return pred


def get_row(ds: Any, index: int) -> tuple[Tensor, Tensor]:
    row = ds[index]
    weights = torch.as_tensor(np.asarray(row["weights"], dtype=np.float32))
    target = torch.as_tensor(np.asarray(row["all_layer_means"], dtype=np.float32))
    return weights, target


def weighted_loss(pred: Tensor, target: Tensor, final_weight: float) -> Tensor:
    depth = pred.shape[0]
    coordinate = torch.linspace(0.0, 1.0, depth, device=pred.device, dtype=pred.dtype)
    weights = 1.0 + (final_weight - 1.0) * coordinate.pow(4)
    layer_mse = (pred - target).square().mean(dim=1)
    return (weights * layer_mse).mean()


@torch.inference_mode()
def evaluate(model: TrajectoryClosure, ds: Any, indices: list[int]) -> dict[str, Any]:
    model.eval()
    pred_rows: list[np.ndarray] = []
    base_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    started = time.perf_counter()
    for index in indices:
        weights, target = get_row(ds, index)
        pred, base = model(weights, return_base=True)
        pred_rows.append(pred.cpu().numpy())
        base_rows.append(base.cpu().numpy())
        target_rows.append(target.cpu().numpy())
    elapsed = time.perf_counter() - started
    pred_np = np.stack(pred_rows)
    base_np = np.stack(base_rows)
    target_np = np.stack(target_rows)
    return {
        "pred": pred_np,
        "base": base_np,
        "target": target_np,
        "model_final_mse": float(np.mean((pred_np[:, -1] - target_np[:, -1]) ** 2)),
        "model_all_mse": float(np.mean((pred_np - target_np) ** 2)),
        "base_final_mse": float(np.mean((base_np[:, -1] - target_np[:, -1]) ** 2)),
        "base_all_mse": float(np.mean((base_np - target_np) ** 2)),
        "seconds": elapsed,
    }


def fit_affine(train_base: np.ndarray, train_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Per-layer scalar affine calibration; strong cheap control for whether the
    # recurrent network is learning structure rather than a global scale drift.
    depth = train_base.shape[1]
    alpha = np.empty(depth, dtype=np.float64)
    beta = np.empty(depth, dtype=np.float64)
    for layer in range(depth):
        x = train_base[:, layer].reshape(-1).astype(np.float64)
        y = train_target[:, layer].reshape(-1).astype(np.float64)
        design = np.stack((x, np.ones_like(x)), axis=1)
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        alpha[layer], beta[layer] = coef
    return alpha, beta


def apply_affine(base: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return base * alpha[None, :, None] + beta[None, :, None]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mini", "full"), default="mini")
    parser.add_argument("--train-count", type=int, required=True)
    parser.add_argument("--val-count", type=int, required=True)
    parser.add_argument("--hidden", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = Config(
        dataset=args.dataset,
        train_count=args.train_count,
        val_count=args.val_count,
        hidden=args.hidden,
        epochs=args.epochs,
        seed=args.seed,
        output=args.output,
    )
    seed_everything(cfg.seed)
    output = Path(cfg.output)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "training.log"

    def log(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    log(f"config={asdict(cfg)}")
    config_name = None if cfg.dataset == "mini" else "full"
    load_args: dict[str, Any] = {
        "path": DATASET_ID,
        "revision": DATASET_REVISION,
        "split": cfg.dataset,
    }
    if config_name is not None:
        load_args["name"] = config_name
    started = time.perf_counter()
    ds = load_dataset(**load_args)
    log(f"loaded dataset rows={len(ds)} seconds={time.perf_counter() - started:.2f}")
    needed = cfg.train_count + cfg.val_count
    if needed > len(ds):
        raise ValueError(f"requested {needed} rows from dataset of size {len(ds)}")

    rng = np.random.default_rng(cfg.seed)
    permutation = rng.permutation(len(ds)).tolist()
    train_indices = permutation[: cfg.train_count]
    val_indices = permutation[cfg.train_count : needed]
    write_json(output / "split.json", {"train": train_indices, "validation": val_indices})

    model = TrajectoryClosure(cfg.hidden, correction_scale=cfg.correction_scale)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    # Cache baseline arrays once. This also records the exact zero-correction
    # starting point before any parameter update.
    initial_train = evaluate(model, ds, train_indices)
    initial_val = evaluate(model, ds, val_indices)
    alpha, beta = fit_affine(initial_train["base"], initial_train["target"])
    affine_val = apply_affine(initial_val["base"], alpha, beta)
    affine_final_mse = float(
        np.mean((affine_val[:, -1] - initial_val["target"][:, -1]) ** 2)
    )
    affine_all_mse = float(np.mean((affine_val - initial_val["target"]) ** 2))
    log(
        "initial "
        f"base_final={initial_val['base_final_mse']:.9e} "
        f"affine_final={affine_final_mse:.9e} "
        f"base_all={initial_val['base_all_mse']:.9e} "
        f"affine_all={affine_all_mse:.9e}"
    )

    best_final = math.inf
    best_epoch = -1
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, float]] = []

    for epoch in range(cfg.epochs):
        model.train()
        order = train_indices.copy()
        random.Random(cfg.seed + 1009 * epoch).shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        epoch_started = time.perf_counter()
        for step, index in enumerate(order, start=1):
            weights, target = get_row(ds, index)
            pred = model(weights)
            assert isinstance(pred, Tensor)
            loss = weighted_loss(pred, target, cfg.final_weight) / cfg.grad_accum
            loss.backward()
            running += float(loss.detach()) * cfg.grad_accum
            if step % cfg.grad_accum == 0 or step == len(order):
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        validation = evaluate(model, ds, val_indices)
        epoch_record = {
            "epoch": float(epoch + 1),
            "train_weighted_loss": running / len(order),
            "val_final_mse": validation["model_final_mse"],
            "val_all_mse": validation["model_all_mse"],
            "seconds": time.perf_counter() - epoch_started,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_record)
        log(
            f"epoch={epoch + 1:03d} train={epoch_record['train_weighted_loss']:.9e} "
            f"val_final={epoch_record['val_final_mse']:.9e} "
            f"val_all={epoch_record['val_all_mse']:.9e} "
            f"lr={epoch_record['learning_rate']:.3e} seconds={epoch_record['seconds']:.2f}"
        )
        if validation["model_final_mse"] < best_final:
            best_final = validation["model_final_mse"]
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    final_train = evaluate(model, ds, train_indices)
    final_val = evaluate(model, ds, val_indices)
    torch.save(
        {
            "config": asdict(cfg),
            "model_state": best_state,
            "alpha": alpha,
            "beta": beta,
        },
        output / "trajectory_closure.pt",
    )
    np.savez_compressed(
        output / "validation_predictions.npz",
        prediction=final_val["pred"],
        baseline=final_val["base"],
        affine=affine_val,
        target=final_val["target"],
        indices=np.asarray(val_indices),
    )
    metrics = {
        "config": asdict(cfg),
        "best_epoch": best_epoch,
        "baseline": {
            "validation_final_mse": initial_val["base_final_mse"],
            "validation_all_mse": initial_val["base_all_mse"],
        },
        "affine": {
            "validation_final_mse": affine_final_mse,
            "validation_all_mse": affine_all_mse,
        },
        "model": {
            "training_final_mse": final_train["model_final_mse"],
            "training_all_mse": final_train["model_all_mse"],
            "validation_final_mse": final_val["model_final_mse"],
            "validation_all_mse": final_val["model_all_mse"],
        },
        "improvement_over_baseline": initial_val["base_final_mse"]
        / final_val["model_final_mse"],
        "improvement_over_affine": affine_final_mse / final_val["model_final_mse"],
        "history": history,
        "total_seconds": time.perf_counter() - started,
    }
    write_json(output / "metrics.json", metrics)
    log(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
