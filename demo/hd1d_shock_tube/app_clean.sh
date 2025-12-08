#!/usr/bin/env bash
# Description: Cleaner script for the MISO application

# Exit on error / undefined variable
set -eu

# Define application directory
APP_DIR=$(cd "$(dirname "$0")" && pwd)

# Run command
set -x
rm -rf "${APP_DIR}"/build "${APP_DIR}"/data_{x,y,z}
