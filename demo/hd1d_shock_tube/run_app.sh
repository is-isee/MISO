#!/usr/bin/env bash
# Description: Launcher script for the MISO application

# Exit on error / undefined variable
set -eu

# Number of processes
NUM_PROCS=1

# Define application name
APP_DIR=$(cd "$(dirname "$0")" && pwd)
APP_NAME="${APP_DIR}/build/$(basename "${APP_DIR}")"

# Load environment variables
set -a
source "${APP_DIR}/openmpi_config.env"
set +a

# Run command
set -x
mpiexec -np ${NUM_PROCS} "${APP_NAME}"
