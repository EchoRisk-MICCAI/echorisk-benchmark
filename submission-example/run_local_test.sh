#!/usr/bin/env bash
# run_local_test.sh — run the reference container against the local fixture
# using the exact invocation pattern the evaluation server uses (see
# docs/HARNESS_SPEC.md §1), then print the resulting predictions.csv.
#
# Usage:
#   ./build.sh 1 && ./run_local_test.sh 1
#
# Note: this defaults to --gpus '"device=0"', matching the real evaluation
# invocation exactly. If your local machine has no GPU (e.g. a laptop used
# only for plumbing checks, not real inference), set NO_GPU=1 to drop the
# flag. Do not rely on NO_GPU as your only test: the real evaluation
# server always passes a GPU, and code that only runs on CPU may still
# fail under --gpus on the actual submission.

set -euo pipefail

TASK_ID="${1:?Usage: ./run_local_test.sh <task_id: 1|2|3>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_TAG="echorisk-reference-task${TASK_ID}:dev"
OUTPUT_DIR="${SCRIPT_DIR}/scratch_output"

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

GPU_FLAGS=(--gpus '"device=0"')
if [[ "${NO_GPU:-0}" == "1" ]]; then
  echo "NO_GPU=1 set: running without --gpus (plumbing check only, not a real evaluation test)."
  GPU_FLAGS=()
fi

echo "Running ${IMAGE_TAG} against fixture data..."
docker run --rm \
  "${GPU_FLAGS[@]}" \
  --network none \
  -v "${SCRIPT_DIR}/fixture:/input:ro" \
  -v "${OUTPUT_DIR}:/output" \
  "${IMAGE_TAG}"

echo
echo "Container exited 0. Output:"
echo "----------------------------------------"
cat "${OUTPUT_DIR}/predictions.csv"
echo "----------------------------------------"

N_MANIFEST_ROWS=$(($(awk 'END{print NR}' "${SCRIPT_DIR}/fixture/manifest.csv") - 1))
N_OUTPUT_ROWS=$(($(awk 'END{print NR}' "${OUTPUT_DIR}/predictions.csv") - 1))

if [[ "${N_OUTPUT_ROWS}" -ne "${N_MANIFEST_ROWS}" ]]; then
  echo "FAIL: manifest has ${N_MANIFEST_ROWS} cases but output has ${N_OUTPUT_ROWS} rows." >&2
  echo "This would fail the official scorer's coverage check (§4 of the spec)." >&2
  exit 1
fi

echo "OK: output covers all ${N_MANIFEST_ROWS} manifest rows."