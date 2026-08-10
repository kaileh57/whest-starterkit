# ruff: noqa: I001
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


INTERESTING = {
    "adjusted_final_layer_score",
    "final_layer_mse",
    "all_layers_mse",
    "mean_flops_used",
    "mean_effective_compute",
    "mean_residual_wall_time_s",
    "flops_used",
    "effective_compute",
    "residual_wall_time_s",
    "score_multiplier",
    "n_failures",
    "failed",
}


def walk(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path, child
            yield from walk(child, path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            path = f"{prefix}[{index}]"
            yield from walk(child, path)


def load_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        raise ValueError("empty output")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Rich/progress output can occasionally precede JSON in non-interactive CI.
        starts = [index for index, char in enumerate(text) if char == "{"]
        for index in starts:
            try:
                return json.loads(text[index:])
            except json.JSONDecodeError:
                continue
        raise


def summarize(name: str, path: Path) -> str:
    try:
        data = load_json(path)
    except Exception as exc:  # preserve broken outputs as useful diagnostics
        return f"{name}: unreadable JSON ({type(exc).__name__}: {exc})"

    found: dict[str, Any] = {}
    for key_path, value in walk(data):
        leaf = key_path.rsplit(".", 1)[-1]
        if leaf in INTERESTING and not isinstance(value, (dict, list)):
            found.setdefault(leaf, value)

    priority = [
        "adjusted_final_layer_score",
        "final_layer_mse",
        "all_layers_mse",
        "mean_effective_compute",
        "mean_flops_used",
        "mean_residual_wall_time_s",
        "n_failures",
    ]
    fields = [f"{key}={found[key]}" for key in priority if key in found]
    if not fields:
        top_keys = sorted(data.keys()) if isinstance(data, dict) else [type(data).__name__]
        fields = [f"top_keys={top_keys}"]
    return f"{name}: " + " ".join(fields)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name", nargs="?")
    parser.add_argument("path", nargs="?", type=Path)
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()

    if args.directory is not None:
        for path in sorted(args.directory.glob("*.json")):
            print(summarize(path.stem, path))
        return
    if args.name is None or args.path is None:
        parser.error("provide NAME PATH, or --directory DIR")
    print(summarize(args.name, args.path))


if __name__ == "__main__":
    main()
