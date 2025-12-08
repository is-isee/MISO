#!/usr/bin/env bash
# Description: Launcher script for the MISO application

# Exit on error / undefined variable
set -eu

# Number of processes
NUM_PROCS=1

# Define application name
APP_DIR=$(cd "$(dirname "$0")" && pwd)

# Build application
cmake -B "${APP_DIR}/build" -S "${APP_DIR}"
cmake --build "${APP_DIR}/build"

# Define application name
APP_NAME="${APP_DIR}/build/$(basename "${APP_DIR}")"
if [[ ! -f "${APP_NAME}" ]]; then
    echo "Error: Application binary not found at ${APP_NAME}"
    exit 1
fi

# Load environment variables
set -a
source "${APP_DIR}"/../shared/openmpi_config.env
set +a

# Run command
set -x
mpiexec -np ${NUM_PROCS} "${APP_NAME}"
