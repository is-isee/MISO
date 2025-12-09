#!/usr/bin/env bash
# Description: Draw figures of the MISO application
# Note: pymiso[vis] needs to be installed

# Exit on error / undefined variable
set -eu

# Define directory of this script
THIS_DIR=$(cd "$(dirname "$0")" && pwd)

# List of demo applications
source "${THIS_DIR}"/shared/app_list.sh

# Run commands
set -x
for APP_NAME in "${APP_LIST[@]}"; do
  APP_DIR="${THIS_DIR}"/"${APP_NAME}"
  python "${APP_DIR}"/plot_data.py
done
