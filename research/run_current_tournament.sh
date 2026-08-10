#!/usr/bin/env bash
set -u -o pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
rm -rf research/results research/vendor
mkdir -p research/results research/vendor

python - <<'PY'
from importlib.metadata import version
expected = {"whestbench": "0.14.0", "flopscope": "0.10.0"}
for package, wanted in expected.items():
    got = version(package)
    print(f"{package}={got}", flush=True)
    if got != wanted:
        raise SystemExit(f"expected {package}=={wanted}, got {got}")
PY

curl --fail --location --retry 4 --retry-delay 2 \
  https://raw.githubusercontent.com/ascender1729/whestbench-cumulant-propagation/c6f87fd1e12634447a452f73ebc136c43bf050d5/estimator.py \
  --output research/vendor/ascender_k3.py
python research/port_ascender.py \
  research/vendor/ascender_k3.py research/vendor/ascender_augops_debug.py --expose

DATASET='hf://aicrowd/arc-whestbench-public-2026@v1-phase1'
N_MLPS="${N_MLPS:-3}"

run_one() {
  local name="$1"
  local estimator="$2"
  local class_name="${3:-Estimator}"
  local count="${4:-$N_MLPS}"
  local out="research/results/${name}.json"
  local log="research/results/${name}.log"

  echo "===== ${name}: validate =====" | tee "$log"
  set +e
  whest validate --estimator "$estimator" --class "$class_name" --format json \
    >>"$log" 2>&1
  local validate_status=$?
  set -e
  if [[ $validate_status -ne 0 ]]; then
    echo "validation failed for ${name}: ${validate_status}" | tee -a "$log"
    return 0
  fi

  echo "===== ${name}: public mini (${count} MLPs) =====" | tee -a "$log"
  set +e
  whest run \
    --estimator "$estimator" \
    --class "$class_name" \
    --dataset "$DATASET" \
    --split mini \
    --n-mlps "$count" \
    --runner local \
    --max-threads 2 \
    --wall-time-limit 60 \
    --format json \
    --debug \
    >"$out" 2>>"$log"
  local status=$?
  set -e
  echo "exit_status=${status}" | tee -a "$log"
  if [[ -s "$out" ]]; then
    python research/summarize_result.py "$name" "$out" | tee -a "$log"
  fi
}

# Execute the current-backend K3 port with every masked exception exposed.
run_one ascender_augops_debug research/vendor/ascender_augops_debug.py Estimator 1

# GL4 has already converged to the GL8/GL12 result; broaden it to five real
# mini MLPs while the K3 port is being exercised.
run_one exact_gaussian_gl4 research/estimators/exact_gaussian.py EstimatorGL4 5

python research/summarize_result.py --directory research/results \
  | tee research/results/summary.txt
