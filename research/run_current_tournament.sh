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

python - <<'PY'
from pathlib import Path

base_path = Path("research/vendor/ascender_k3.py")
src = base_path.read_text(encoding="utf-8")

# flopscope 0.10 arrays no longer implement the real-numpy array-function path
# for mutation. The public estimator swallowed this and returned all zeros.
ported = src.replace("np.fill_diagonal(", "fnp.fill_diagonal(")

outer_mask = "        except Exception:\n            return fnp.zeros((depth, width), dtype=fnp.float64)"
outer_expose = (
    "        except Exception as _research_exc:\n"
    "            import traceback as _research_tb\n"
    "            print('ASCENDER_OUTER_FAILURE:', repr(_research_exc), flush=True)\n"
    "            _research_tb.print_exc()\n"
    "            raise"
)
if outer_mask not in ported:
    raise SystemExit("could not find final zero-fallback mask")
ported_exposed = ported.rsplit(outer_mask, 1)[0] + outer_expose + ported.rsplit(outer_mask, 1)[1]
Path("research/vendor/ascender_port_fill.py").write_text(ported_exposed, encoding="utf-8")

# The K3 call has a second mask that falls back to covariance. Expose it in a
# separate copy so the next unsupported operation is visible.
inner_mask = "                except Exception:\n                    means = None"
inner_expose = (
    "                except Exception as _k3_exc:\n"
    "                    import traceback as _k3_tb\n"
    "                    print('ASCENDER_K3_FAILURE:', repr(_k3_exc), flush=True)\n"
    "                    _k3_tb.print_exc()\n"
    "                    raise"
)
if inner_mask not in ported_exposed:
    raise SystemExit("could not find inner k3 fallback mask")
core_debug = ported_exposed.replace(inner_mask, inner_expose, 1)
Path("research/vendor/ascender_core_debug.py").write_text(core_debug, encoding="utf-8")
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

# First establish whether the one-line current-backend port recovers a valid
# fallback, then expose the actual K3 failure behind its inner mask.
run_one ascender_port_fill research/vendor/ascender_port_fill.py Estimator 1
run_one ascender_core_debug research/vendor/ascender_core_debug.py Estimator 1

# Independent exact bivariate-Gaussian closure sweep. A single real MLP is
# enough to reject broken or over-budget variants before broader evaluation.
run_one exact_gaussian_gl4 research/estimators/exact_gaussian.py EstimatorGL4 1
run_one exact_gaussian_gl8 research/estimators/exact_gaussian.py EstimatorGL8 1
run_one exact_gaussian_gl12 research/estimators/exact_gaussian.py EstimatorGL12 1

python research/summarize_result.py --directory research/results \
  | tee research/results/summary.txt
