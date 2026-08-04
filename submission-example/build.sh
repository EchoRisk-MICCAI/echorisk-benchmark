#!/usr/bin/env bash
# build.sh — build the reference image for a given task.
#
# Usage:
#   ./build.sh 1    # builds a Task 1 image
#   ./build.sh 2    # builds a Task 2 image
#   ./build.sh 3    # builds a Task 3 image

set -euo pipefail

TASK_ID="${1:?Usage: ./build.sh <task_id: 1|2|3>}"

if [[ ! "${TASK_ID}" =~ ^[123]$ ]]; then
  echo "error: task_id must be 1, 2, or 3; got '${TASK_ID}'." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_TAG="echorisk-reference-task${TASK_ID}:dev"

echo "Building ${IMAGE_TAG} (TASK_ID=${TASK_ID})..."
docker build \
  --build-arg TASK_ID="${TASK_ID}" \
  -t "${IMAGE_TAG}" \
  "${SCRIPT_DIR}"

echo "Built ${IMAGE_TAG}."