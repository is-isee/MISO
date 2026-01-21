#!/usr/bin/env bash
# Description: Clean binaries and output of MISO

# Exit on error / undefined variable
set -eu

# Define directory of this script
THIS_DIR=$(cd "$(dirname "$0")" && pwd)

# Root directory
MISO_ROOT="${THIS_DIR}/../.."

# Source and binary directories
MISO_SRC="${MISO_ROOT}/core"
MISO_BIN="${MISO_SRC}/build"

# Run commands
set -x
rm -rf "${MISO_BIN}"
"${MISO_ROOT}"/demo/demo_clean.sh
