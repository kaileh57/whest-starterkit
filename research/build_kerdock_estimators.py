#!/usr/bin/env python3
"""Build Kerdock/MUB cubature variants of Oishi1029's MIT sweep estimator.

The upstream propagation implementation is preserved. We replace only the
Gaussian input draw/whitening stage with complete mutually-unbiased bases and,
for selected variants, disable the frozen lead-block mask so all remaining
pruning is algebraically exact.

Every generated file is self-contained apart from ``mub_design.py`` in the same
directory. Anchor replacements fail closed if the pinned upstream changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_BASIS_COUNTS = (8, 12, 16, 20, 24, 32, 48, 64, 96, 129)
IMPORT_ANCHOR = (
    "from whestbench import BaseEstimator, SetupContext\n"
    "from whestbench.domain import MLP\n"
)
INIT_ANCHOR = """    def __init__(self) -> None:
        self._setup_rng = None
"""
SETUP_ANCHOR = """        self._setup_rng = fnp.random.default_rng(ctx.seed)
"""
COUNT_ANCHOR = """        k = _sample_count(int(budget), width, depth)
"""
SAMPLE_ANCHOR = """        rng = fnp.random.default_rng(mlp.seed)
        half = k // 2
        xh = rng.standard_normal((half, width), dtype=fnp.float32)

        w0 = fnp.asarray(mlp.weights[0], dtype=fnp.float32)
        w0f = self._fused_first_layer(xh, w0, k, width)
"""
TAIL_ANCHOR = "_MIN_TAIL_LAYERS = 3\n"
DOC_ANCHOR = (
    '"""Whitened + antithetic MC with two-sided lead-block pruning of the forward pass."""'
)


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one {label} anchor, found {count}")
    return text.replace(old, new, 1)


def build(source: str, *, num_bases: int, exact_tail: bool) -> str:
    if not 1 <= int(num_bases) <= 129:
        raise ValueError("num_bases must lie in [1,129]")

    text = source
    text = _replace_once(
        text,
        IMPORT_ANCHOR,
        IMPORT_ANCHOR + "from mub_design import build_half_design\n",
        "import",
    )
    text = _replace_once(
        text,
        INIT_ANCHOR,
        """    def __init__(self) -> None:
        self._setup_rng = None
        self._design = None
""",
        "init",
    )
    text = _replace_once(
        text,
        SETUP_ANCHOR,
        SETUP_ANCHOR
        + f"""        # Constructed before FLOP tracking. Each complete basis has exact
        # covariance; the full 129-basis union is an antipodal spherical 5-design.
        # Rows already include E[Chi_256], analytically integrating Gaussian radius.
        self._design = fnp.asarray(
            build_half_design(
                seed=ctx.seed, shuffle=True, num_bases={int(num_bases)}
            ),
            dtype=fnp.float32,
        )
""",
        "setup",
    )
    text = _replace_once(
        text,
        COUNT_ANCHOR,
        """        if width != 256 or self._design is None:
            raise ValueError("Kerdock estimator requires width 256 and completed setup")
        k = int(2 * self._design.shape[0])
""",
        "sample count",
    )
    text = _replace_once(
        text,
        SAMPLE_ANCHOR,
        """        # One representative from every antipodal pair. _layer0 constructs
        # the exact negatives algebraically, so no mirrored matmul is performed.
        half = k // 2
        xh = self._design

        w0 = fnp.asarray(mlp.weights[0], dtype=fnp.float32)
        w0f = w0
""",
        "ensemble",
    )

    if exact_tail:
        # This forces the upstream legacy path for depth 32. That path still uses
        # exact row pruning, but never freezes a mask inferred from 1,024 nodes and
        # never classifies final-layer columns from the lead block.
        text = _replace_once(
            text,
            TAIL_ANCHOR,
            "_MIN_TAIL_LAYERS = 10_000\n",
            "tail-mask constant",
        )
        mode = "exact-tail"
    else:
        mode = "lead-mask"

    degree = 5 if int(num_bases) == 129 else 3
    title = (
        f"Kerdock/MUB {int(num_bases)}-basis antipodal spherical-{degree}-design "
        f"sweep ({mode})"
    )
    text = _replace_once(text, DOC_ANCHOR, f'"""{title}."""', "class docstring")

    provenance = (
        "# GENERATED RESEARCH VARIANT. Upstream propagation implementation: "
        "Oishi1029/arc-whestbench-2026 (MIT), commit "
        "230a3acae4508ff62dbf57d8c13e534c32583e0a.\n"
        "# Cubature construction and patching: kaileh57/whest-starterkit.\n\n"
    )
    return provenance + text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--basis-counts",
        default=",".join(str(x) for x in DEFAULT_BASIS_COUNTS),
        help="comma-separated complete-basis counts",
    )
    args = parser.parse_args()

    counts = tuple(int(part.strip()) for part in args.basis_counts.split(",") if part.strip())
    if not counts:
        raise ValueError("at least one basis count is required")
    if len(set(counts)) != len(counts):
        raise ValueError("basis counts must be unique")

    source = args.source.read_text(encoding="utf-8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for num_bases in counts:
        for exact_tail in (False, True):
            suffix = "exact" if exact_tail else "mask"
            path = args.output_dir / f"kerdock_b{num_bases:03d}_{suffix}.py"
            path.write_text(
                build(source, num_bases=num_bases, exact_tail=exact_tail),
                encoding="utf-8",
            )

    helper = Path(__file__).with_name("mub_design.py")
    (args.output_dir / "mub_design.py").write_bytes(helper.read_bytes())
    print(
        f"built {2 * len(counts)} variants in {args.output_dir}: "
        + ", ".join(str(x) for x in counts)
    )


if __name__ == "__main__":
    main()
