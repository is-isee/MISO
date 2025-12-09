#!/usr/bin/env bash
# Description: Clean binaries and output of MISO

# Exit on error / undefined variable
set -eu

# Define directory of this script
THIS_DIR=$(cd "$(dirname "$0")" && pwd)

# Run commands
set -x
rm -rf "${THIS_DIR}"/core/build
"${THIS_DIR}"/demo/demo_clean.sh
