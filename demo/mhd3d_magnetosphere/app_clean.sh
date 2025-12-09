#!/usr/bin/env bash
# Description: Clean output data of the MISO application

# Exit on error / undefined variable
set -eu

# Define directory of this script
THIS_DIR=$(cd "$(dirname "$0")" && pwd)

# Run command
set -x
rm -rf "${THIS_DIR}"/build "${THIS_DIR}"/data
