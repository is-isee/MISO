#!/usr/bin/env bash
# Description: Run the MISO application

# Exit on error / undefined variable
set -eu

# Define directory of this script
THIS_DIR=$(cd "$(dirname "$0")" && pwd)

# List of demo applications
source "${THIS_DIR}"/shared/app_list.sh

# Ask user for confirmation
echo "This process may take a long time. Do you really want to run it? (y/n)"
read -r answer
case "$answer" in
  [Yy]*)
    echo "Starting execution..."
    ;;
  *)
    echo "Operation cancelled."
    exit 0
    ;;
esac

# Run commands
set -x
for APP_NAME in "${APP_LIST[@]}"; do
  APP_DIR="${THIS_DIR}"/"${APP_NAME}"
  "${APP_DIR}"/app_run.sh
done
