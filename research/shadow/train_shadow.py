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
    learning_rate: float = 1.2e-3
    weight_decay: float = 2.0e-5
    grad_accum: int = 4
    final_weight: float = 48.0
    defect_scale_mean: float = 0.06
    defect_scale_variance: float = 0.16
    teacher_weight: float = 0.12


@dataclass
class Record:
    weights: Tensor
    target: Tensor
    base_mean: Tensor
    base_variance: Tensor
    gain: Tensor
    phi: Tensor
    sigma_pre: Tensor
    variance_pre: Tensor
    local: Tensor
    teacher_defect: Tensor


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))


def relu_marginal(mu: Tensor, var: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    var = var.clamp_min(1.0e-12)
    sigma = var.sqrt()
    a = mu / sigma
    phi = torch.exp(-0.5 * a.square()) / SQRT_2PI
    gain = 0.5 * (1.0 + torch.erf(a / SQRT_2))
    mean = mu * gain + sigma * phi
    second = (mu.square() + var) * gain + mu * sigma * phi
    post_var = (second - mean.square()).clamp_min(1.0e-12)
    return mean, post_var, gain, phi


@torch.inference_mode()
def build_record(row: dict[str, Any]) -> Record:
    weights = torch.as_tensor(np.asarray(row["weights"], dtype=np.float32))
    target = torch.as_tensor(np.asarray(row["all_layer_means"], dtype=np.float32))
    depth, width, _ = weights.shape
    mean = torch.zeros(width, dtype=torch.float32)
    covariance = torch.eye(width, dtype=torch.float32)

    base_means: list[Tensor] = []
    base_variances: list[Tensor] = []
    gains: list[Tensor] = []
    phis: list[Tensor] = []
    sigmas: list[Tensor] = []
    variances_pre: list[Tensor] = []
    locals_: list[Tensor] = []

    for layer, weight in enumerate(weights):
        mu = mean @ weight
        covariance_pre = weight.T @ covariance @ weight
        covariance_pre = 0.5 * (covariance_pre + covariance_pre.T)
        variance_pre = torch.diagonal(covariance_pre).clamp_min(1.0e-12)
        mean_post, variance_post, gain, phi = relu_marginal(mu, variance_pre)

        # First-order Gaussian covariance closure, exactly matching the starter
        # covariance baseline while preserving the exact marginal diagonal.
        covariance = covariance_pre * gain[:, None] * gain[None, :]
        covariance.diagonal().copy_(variance_post)
        covariance = 0.5 * (covariance + covariance.T)
        mean = mean_post

        energy = weight.square().sum(dim=0).clamp_min(1.0e-8)
        sigma = variance_pre.sqrt()
        local = torch.stack(
            (
                mu / sigma,
                torch.log(variance_pre),
                mean_post / sigma,
                variance_post / variance_pre,
                weight.sum(dim=0) / SQRT_2,
                energy - 2.0,
                weight.pow(3).sum(dim=0) / energy.pow(1.5),
                weight.pow(4).sum(dim=0) / energy.square(),
                torch.full_like(mu, (layer + 1.0) / depth),
            ),
            dim=1,
        )
        base_means.append(mean_post)
        base_variances.append(variance_post)
        gains.append(gain)
        phis.append(phi)
        sigmas.append(sigma)
        variances_pre.append(variance_pre)
        locals_.append(local)

    base_mean = torch.stack(base_means)
    gain = torch.stack(gains)
    true_error = target - base_mean
    teacher: list[Tensor] = []
    previous = torch.zeros(width)
    for layer, weight in enumerate(weights):
        transported = gain[layer] * (previous @ weight)
        teacher.append(true_error[layer] - transported)
        previous = true_error[layer]

    return Record(
        weights=weights,
        target=target,
        base_mean=base_mean,
        base_variance=torch.stack(base_variances),
        gain=gain,
        phi=torch.stack(phis),
        sigma_pre=torch.stack(sigmas),
        variance_pre=torch.stack(variances_pre),
        local=torch.stack(locals_),
        teacher_defect=torch.stack(teacher),
    )


class ShadowClosure(nn.Module):
    """Learned local defects transported on a frozen covariance trajectory.

    The base covariance chain is never perturbed. Mean and marginal-variance
    errors evolve only through the exact tangent of the Gaussian ReLU moment
    map plus a learned bounded defect. Hidden channels move between independently
    permutable layers through W and W^2 messages, preserving equivariance.
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.hidden = config.hidden
        self.scale_mean = config.defect_scale_mean
        self.scale_variance = config.defect_scale_variance
        feature_dim = 11 + 2 * config.hidden
        trunk = max(48, 4 * config.hidden)
        self.local = nn.Sequential(
            nn.Linear(feature_dim, trunk),
            nn.SiLU(),
            nn.Linear(trunk, trunk),
            nn.SiLU(),
            nn.Linear(trunk, config.hidden + 2),
        )
        last = self.local[-1]
        assert isinstance(last, nn.Linear)
        with torch.no_grad():
            nn.init.normal_(last.weight[: config.hidden], mean=0.0, std=0.015)
            last.bias[: config.hidden].zero_()
            last.weight[config.hidden :].zero_()
            last.bias[config.hidden :].zero_()

    def forward(self, record: Record, *, return_defects: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        depth, width, _ = record.weights.shape
        error_mean = torch.zeros(width, dtype=record.weights.dtype)
        error_variance = torch.zeros_like(error_mean)
        hidden = torch.zeros((width, self.hidden), dtype=record.weights.dtype)
        rows: list[Tensor] = []
        defects: list[Tensor] = []

        for layer in range(depth):
            weight = record.weights[layer]
            weight2 = weight.square()
            gain = record.gain[layer]
            phi = record.phi[layer]
            sigma = record.sigma_pre[layer]
            variance_pre = record.variance_pre[layer]
            base_mean = record.base_mean[layer]

            delta_mu = error_mean @ weight
            delta_variance_pre = error_variance @ weight2
            transported_mean = gain * delta_mu + 0.5 * phi / sigma * delta_variance_pre
            derivative_variance_mu = 2.0 * base_mean * (1.0 - gain)
            derivative_variance_var = gain - base_mean * phi / sigma
            transported_variance = (
                derivative_variance_mu * delta_mu
                + derivative_variance_var * delta_variance_pre
            )

            msg_signed = (weight.T @ hidden) / SQRT_2
            energy = weight2.sum(dim=0).clamp_min(1.0e-8)
            msg_square = (weight2.T @ hidden) / energy[:, None]
            transport_features = torch.stack(
                (
                    transported_mean / sigma,
                    transported_variance / variance_pre,
                ),
                dim=1,
            )
            features = torch.cat(
                (record.local[layer], transport_features, msg_signed, msg_square), dim=1
            )
            raw = self.local(features)
            hidden = torch.tanh(raw[:, : self.hidden])
            defect_mean = self.scale_mean * sigma * torch.tanh(raw[:, self.hidden])
            defect_variance = (
                self.scale_variance
                * variance_pre
                * torch.tanh(raw[:, self.hidden + 1])
            )
            error_mean = transported_mean + defect_mean
            error_variance = transported_variance + defect_variance
            rows.append(base_mean + error_mean)
            defects.append(defect_mean)

        prediction = torch.stack(rows)
        if return_defects:
            return prediction, torch.stack(defects)
        return prediction


def rollout_loss(prediction: Tensor, target: Tensor, final_weight: float) -> Tensor:
    depth = prediction.shape[0]
    coordinate = torch.linspace(0.0, 1.0, depth, dtype=prediction.dtype)
    weights = 1.0 + (final_weight - 1.0) * coordinate.pow(4)
    return (weights * (prediction - target).square().mean(dim=1)).mean()


def fit_affine(records: list[Record]) -> tuple[np.ndarray, np.ndarray]:
    base = np.stack([record.base_mean.numpy() for record in records])
    target = np.stack([record.target.numpy() for record in records])
    depth = base.shape[1]
    alpha = np.empty(depth, dtype=np.float64)
    beta = np.empty(depth, dtype=np.float64)
    for layer in range(depth):
        x = base[:, layer].reshape(-1).astype(np.float64)
        y = target[:, layer].reshape(-1).astype(np.float64)
        design = np.stack((x, np.ones_like(x)), axis=1)
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        alpha[layer], beta[layer] = coef
    return alpha, beta


@torch.inference_mode()
def evaluate(model: ShadowClosure, records: list[Record]) -> dict[str, Any]:
    model.eval()
    prediction = np.stack([model(record).numpy() for record in records])
    target = np.stack([record.target.numpy() for record in records])
    base = np.stack([record.base_mean.numpy() for record in records])
    return {
        "prediction": prediction,
        "target": target,
        "base": base,
        "model_final_mse": float(np.mean((prediction[:, -1] - target[:, -1]) ** 2)),
        "model_all_mse": float(np.mean((prediction - target) ** 2)),
        "base_final_mse": float(np.mean((base[:, -1] - target[:, -1]) ** 2)),
        "base_all_mse": float(np.mean((base - target) ** 2)),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mini", "full"), default="mini")
    parser.add_argument("--train-count", type=int, required=True)
    parser.add_argument("--val-count", type=int, required=True)
    parser.add_argument("--hidden", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = Config(
        dataset=args.dataset,
        train_count=args.train_count,
        val_count=args.val_count,
        hidden=args.hidden,
        epochs=args.epochs,
        seed=args.seed,
        output=args.output,
    )
    seed_everything(config.seed)
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "training.log"

    def log(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    started = time.perf_counter()
    log(f"config={asdict(config)}")
    load_args: dict[str, Any] = {
        "path": DATASET_ID,
        "revision": DATASET_REVISION,
        "split": config.dataset,
    }
    if config.dataset == "full":
        load_args["name"] = "full"
    dataset = load_dataset(**load_args)
    log(f"loaded rows={len(dataset)} seconds={time.perf_counter() - started:.2f}")
    needed = config.train_count + config.val_count
    if needed > len(dataset):
        raise ValueError(f"requested {needed} rows from {len(dataset)}")
    rng = np.random.default_rng(config.seed)
    indices = rng.permutation(len(dataset)).tolist()[:needed]
    train_indices = indices[: config.train_count]
    val_indices = indices[config.train_count :]
    write_json(output / "split.json", {"train": train_indices, "validation": val_indices})

    records: dict[int, Record] = {}
    for position, index in enumerate(indices, start=1):
        records[index] = build_record(dataset[index])
        if position == 1 or position % 10 == 0 or position == len(indices):
            log(
                f"precomputed={position}/{len(indices)} "
                f"seconds={time.perf_counter() - started:.2f}"
            )
    train_records = [records[index] for index in train_indices]
    val_records = [records[index] for index in val_indices]

    model = ShadowClosure(config)
    initial_val = evaluate(model, val_records)
    alpha, beta = fit_affine(train_records)
    val_base = initial_val["base"]
    val_target = initial_val["target"]
    affine = val_base * alpha[None, :, None] + beta[None, :, None]
    affine_final = float(np.mean((affine[:, -1] - val_target[:, -1]) ** 2))
    affine_all = float(np.mean((affine - val_target) ** 2))
    log(
        "initial "
        f"base_final={initial_val['base_final_mse']:.9e} "
        f"affine_final={affine_final:.9e} "
        f"base_all={initial_val['base_all_mse']:.9e} "
        f"affine_all={affine_all:.9e}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    best_final = math.inf
    best_epoch = -1
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, float]] = []

    for epoch in range(config.epochs):
        model.train()
        order = list(range(len(train_records)))
        random.Random(config.seed + epoch * 1009).shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        running_rollout = 0.0
        running_teacher = 0.0
        epoch_started = time.perf_counter()
        for step, position in enumerate(order, start=1):
            record = train_records[position]
            prediction, defect = model(record, return_defects=True)
            roll = rollout_loss(prediction, record.target, config.final_weight)
            teacher = (
                ((defect - record.teacher_defect) / record.sigma_pre)
                .square()
                .mean()
            )
            loss = (roll + config.teacher_weight * teacher) / config.grad_accum
            loss.backward()
            running_rollout += float(roll.detach())
            running_teacher += float(teacher.detach())
            if step % config.grad_accum == 0 or step == len(order):
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        validation = evaluate(model, val_records)
        record_epoch = {
            "epoch": float(epoch + 1),
            "train_rollout": running_rollout / len(order),
            "train_teacher": running_teacher / len(order),
            "validation_final_mse": validation["model_final_mse"],
            "validation_all_mse": validation["model_all_mse"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(record_epoch)
        log(
            f"epoch={epoch + 1:03d} rollout={record_epoch['train_rollout']:.9e} "
            f"teacher={record_epoch['train_teacher']:.9e} "
            f"val_final={record_epoch['validation_final_mse']:.9e} "
            f"val_all={record_epoch['validation_all_mse']:.9e} "
            f"lr={record_epoch['learning_rate']:.3e} seconds={record_epoch['seconds']:.2f}"
        )
        if validation["model_final_mse"] < best_final:
            best_final = validation["model_final_mse"]
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("no checkpoint produced")
    model.load_state_dict(best_state)
    final_train = evaluate(model, train_records)
    final_val = evaluate(model, val_records)
    torch.save(
        {"config": asdict(config), "model_state": best_state, "alpha": alpha, "beta": beta},
        output / "shadow_closure.pt",
    )
    np.savez_compressed(
        output / "validation_predictions.npz",
        prediction=final_val["prediction"],
        baseline=final_val["base"],
        affine=affine,
        target=final_val["target"],
        indices=np.asarray(val_indices),
    )
    metrics = {
        "config": asdict(config),
        "best_epoch": best_epoch,
        "baseline": {
            "validation_final_mse": initial_val["base_final_mse"],
            "validation_all_mse": initial_val["base_all_mse"],
        },
        "affine": {
            "validation_final_mse": affine_final,
            "validation_all_mse": affine_all,
        },
        "model": {
            "training_final_mse": final_train["model_final_mse"],
            "training_all_mse": final_train["model_all_mse"],
            "validation_final_mse": final_val["model_final_mse"],
            "validation_all_mse": final_val["model_all_mse"],
        },
        "improvement_over_baseline": initial_val["base_final_mse"]
        / final_val["model_final_mse"],
        "improvement_over_affine": affine_final / final_val["model_final_mse"],
        "history": history,
        "total_seconds": time.perf_counter() - started,
    }
    write_json(output / "metrics.json", metrics)
    log(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
