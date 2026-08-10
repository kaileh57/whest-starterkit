#!/usr/bin/env bash
set -u -o pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
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

# The public file deliberately masks every final failure by returning zeros.
# Make research-only copies that expose the traceback and test the most likely
# v0.10 compatibility change: keep the grader's arrays in flopscope.numpy.
python - <<'PY'
from pathlib import Path

base_path = Path("research/vendor/ascender_k3.py")
src = base_path.read_text(encoding="utf-8")
mask = "        except Exception:\n            return fnp.zeros((depth, width), dtype=fnp.float64)"
expose = (
    "        except Exception as _research_exc:\n"
    "            import traceback as _research_tb\n"
    "            print('ASCENDER_CURRENT_FAILURE:', repr(_research_exc), flush=True)\n"
    "            _research_tb.print_exc()\n"
    "            raise"
)
if mask not in src:
    raise SystemExit("could not find final zero-fallback mask")
debug = src.rsplit(mask, 1)[0] + expose + src.rsplit(mask, 1)[1]
Path("research/vendor/ascender_debug.py").write_text(debug, encoding="utf-8")

old_weights = "Ws = [np.asarray(w, dtype=np.float64) for w in mlp.weights]"
new_weights = "Ws = [fnp.asarray(w, dtype=fnp.float64) for w in mlp.weights]"
if old_weights not in debug:
    raise SystemExit("could not find weight conversion")
Path("research/vendor/ascender_fnp_weights.py").write_text(
    debug.replace(old_weights, new_weights, 1), encoding="utf-8"
)
PY

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

# Diagnose the current K3 port before spending time on broad runs.
run_one ascender_debug research/vendor/ascender_debug.py Estimator 1
run_one ascender_fnp_weights research/vendor/ascender_fnp_weights.py Estimator 1

# Cheap reference points under the exact current accounting model.
run_one mean examples/02_mean_propagation.py
run_one covariance examples/03_covariance_propagation.py
# Public file as published, including its zero fallback.
run_one ascender_k3 research/vendor/ascender_k3.py

python research/summarize_result.py --directory research/results \
  | tee research/results/summary.txt
