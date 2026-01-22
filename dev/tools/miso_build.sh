#!/usr/bin/env bash
# Description: Build tests and demos of MISO

# Exit on error / undefined variable
set -eu

# Parse arguments
MISO_USE_CUDA=OFF
if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [--cpu | --cuda]" >&2
  exit 1
fi
if [[ $# -eq 1 ]]; then
  case "$1" in
    --cpu)
      MISO_USE_CUDA=OFF
      ;;
    --cuda)
      MISO_USE_CUDA=ON
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 [--cpu | --cuda]" >&2
      exit 1
      ;;
  esac
fi

# Define directory of this script
THIS_DIR=$(cd "$(dirname "$0")" && pwd)

# Root directory
MISO_ROOT=$(cd "${THIS_DIR}/../.." && pwd)

# Source and binary directories
MISO_SRC="${MISO_ROOT}"/miso
MISO_BIN="${MISO_SRC}"/build

# Run commands
set -x
cmake -B "${MISO_BIN}" -S "${MISO_SRC}" -DMISO_USE_CUDA=${MISO_USE_CUDA}
cmake --build "${MISO_BIN}" -j
if [[ $# -ge 1 ]]; then
  "${MISO_ROOT}"/demo/demo_build.sh "$1"
else
  "${MISO_ROOT}"/demo/demo_build.sh
fi
