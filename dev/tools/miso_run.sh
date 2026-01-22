#!/usr/bin/env bash
# Description: Run tests and demos of MISO

# Exit on error / undefined variable
set -eu

# Define directory of this script
THIS_DIR=$(cd "$(dirname "$0")" && pwd)

# Root directory
MISO_ROOT=$(cd "${THIS_DIR}/../.." && pwd)

# Source and binary directories
MISO_SRC="${MISO_ROOT}"/miso
MISO_BIN="${MISO_SRC}"/build

# Run commands
set -x
cmake --build "${MISO_BIN}" --target test
"${MISO_ROOT}"/demo/demo_run.sh
