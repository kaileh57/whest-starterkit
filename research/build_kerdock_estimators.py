#!/usr/bin/env python3
"""Build Kerdock spherical-design variants of Oishi1029's MIT sweep estimator.

The source estimator is downloaded by the research workflow from the pinned
upstream commit.  We preserve the complete propagation/pruning implementation
and replace only the input ensemble and whitening stage:

* 66,048 equal-weight nodes from the exact R^256 Kerdock/MUB spherical 5-design;
* exact analytic Gaussian radial mean E[Chi_256], already folded into the nodes;
* optional per-MLP Haar rotation fused into W_0.

This script deliberately fails closed if any source anchor changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path


IMPORT_ANCHOR = "from whestbench import BaseEstimator, MLP, SetupContext\n"
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


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one {label} anchor, found {count}")
    return text.replace(old, new, 1)


def build(source: str, *, haar: bool) -> str:
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
        + """        # Fixed design construction happens before FLOP tracking and uses no
        # MLP-specific information.  Every row already includes E[Chi_256].
        self._design = fnp.asarray(
            build_half_design(seed=ctx.seed, shuffle=False), dtype=fnp.float32
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

    if haar:
        sample = """        # The exact design is deterministic; one seeded Haar rotation turns it
        # into randomized cubature without changing its degree-five exactness.  Fuse the
        # rotation into W_0 so the 66,048 design rows are never explicitly rotated.
        rng = fnp.random.default_rng(mlp.seed)
        half = k // 2
        xh = self._design

        w0 = fnp.asarray(mlp.weights[0], dtype=fnp.float32)
        raw_q = rng.standard_normal((width, width), dtype=fnp.float32)
        q, r = fnp.linalg.qr(raw_q, mode="reduced")
        diagonal = fnp.diag(r)
        signs = fnp.where(diagonal >= 0.0, 1.0, -1.0)
        q = q * signs
        w0f = q @ w0
"""
        title = "Kerdock/MUB spherical-5-design sweep with seeded Haar rotation"
    else:
        sample = """        # One representative of every antipodal pair.  _layer0 constructs
        # the exact negatives algebraically, as in the upstream sweep estimator.
        half = k // 2
        xh = self._design

        w0 = fnp.asarray(mlp.weights[0], dtype=fnp.float32)
        w0f = w0
"""
        title = "Fixed-orientation Kerdock/MUB spherical-5-design sweep"

    text = _replace_once(text, SAMPLE_ANCHOR, sample, "ensemble")
    text = text.replace(
        '"""Whitened + antithetic MC with two-sided lead-block pruning of the forward pass."""',
        f'"""{title}."""',
        1,
    )
    provenance = (
        "# GENERATED RESEARCH VARIANT. Upstream propagation implementation: "
        "Oishi1029/arc-whestbench-2026 (MIT).\n"
        "# Input cubature and patching: kaileh57/whest-starterkit research branch.\n\n"
    )
    return provenance + text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    source = args.source.read_text(encoding="utf-8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "kerdock_fixed.py").write_text(
        build(source, haar=False), encoding="utf-8"
    )
    (args.output_dir / "kerdock_haar.py").write_text(
        build(source, haar=True), encoding="utf-8"
    )
    helper = Path(__file__).with_name("mub_design.py")
    (args.output_dir / "mub_design.py").write_bytes(helper.read_bytes())
    print(f"built variants in {args.output_dir}")


if __name__ == "__main__":
    main()
