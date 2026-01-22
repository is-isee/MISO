#!/usr/bin/env bash
# Description: Run the MISO application

# Exit on error / undefined variable
set -eu

# Parse arguments
USE_CUDA=OFF
if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [--cpu | --cuda]" >&2
  exit 1
fi
if [[ $# -eq 1 ]]; then
  case "$1" in
    --cpu)
      USE_CUDA=OFF
      ;;
    --cuda)
      USE_CUDA=ON
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

# List of demo applications
source "${THIS_DIR}"/shared/app_list.sh

# Run commands
set -x
for APP_NAME in "${APP_LIST[@]}"; do
  APP_DIR="${THIS_DIR}"/"${APP_NAME}"
  APP_BIN="${APP_DIR}"/build
  cmake -B "${APP_BIN}" -S "${APP_DIR}" -DUSE_CUDA="${USE_CUDA}"
  cmake --build "${APP_BIN}" -j
done
