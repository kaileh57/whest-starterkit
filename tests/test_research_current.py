from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def _read_json(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    start = text.find("{")
    assert start >= 0, f"no JSON object in {path}"
    return json.loads(text[start:])


def test_current_harness_research_tournament() -> None:
    """Temporary draft-PR executor for the v0.14/v0.10 research loop."""
    env = os.environ.copy()
    env.setdefault("N_MLPS", "3")
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("OPENBLAS_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    env.setdefault("NUMEXPR_NUM_THREADS", "2")
    env.setdefault("WHEST_SKIP_HARDWARE_FALLBACK_PROBES", "1")
    subprocess.run(
        ["bash", "research/run_current_tournament.sh"],
        check=True,
        timeout=12 * 60,
        env=env,
    )

    results = Path("research/results")
    assert (results / "summary.txt").is_file()
    exact = _read_json(results / "exact_gaussian_gl4.json")
    assert exact.get("ok") is True
    # The K3 debug variant may intentionally fail while the compatibility port
    # is being developed, but it must leave a report rather than silently zero.
    assert (results / "ascender_augops_debug.json").is_file()
