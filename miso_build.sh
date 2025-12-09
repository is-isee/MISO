#!/usr/bin/env bash
# Description: Build tests and demos of MISO

# Exit on error / undefined variable
set -eu

# Define directory of this script
THIS_DIR=$(cd "$(dirname "$0")" && pwd)

# Source and binary directories: MISO core
MISO_SRC="${THIS_DIR}"/core
MISO_BIN="${MISO_SRC}"/build

# Run commands
set -x
cmake -B "${MISO_BIN}" -S "${MISO_SRC}"
cmake --build "${MISO_BIN}" -j
"${THIS_DIR}"/demo/demo_build.sh
