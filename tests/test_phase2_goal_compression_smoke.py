from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_phase2_goal_compression_smoke(tmp_path: Path) -> None:
    """Run a small unseen-network ablation and expose its aggregate in CI logs."""
    output = tmp_path / "goal_compression.json"
    command = [
        sys.executable,
        "research/phase2_goal_compression.py",
        "--width",
        "24",
        "--depth",
        "16",
        "--seeds",
        "2",
        "--seed-offset",
        "91000",
        "--ranks",
        "6,12,24",
        "--qmc-exp",
        "12",
        "--output",
        str(output),
    ]
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
        timeout=1_200,
    )
    print(completed.stdout)
    print(completed.stderr)
    assert completed.returncode == 0
    report = json.loads(output.read_text())
    aggregate = report["aggregate"]
    print("PHASE2_GOAL_COMPRESSION_RESULT=" + json.dumps(aggregate, sort_keys=True))
    assert aggregate["cases"] == 2
    for rank in ("6", "12", "24"):
        row = aggregate["ranks"][rank]
        assert row["goal_fidelity_geomean"] >= 0.0
        assert row["magnitude_fidelity_geomean"] >= 0.0
