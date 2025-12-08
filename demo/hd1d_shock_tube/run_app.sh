#!/usr/bin/env bash

# Error handling
set -eu

# Number of processes
NUM_PROCS=1

# Define application name
APP_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_NAME=$(basename "${APP_DIR}")
APP_NAME="${APP_DIR}/build/${PROJECT_NAME}"

# Load environment variables
set -a
source "${APP_DIR}/openmpi_config.env"
set +a

# Run the application
CMD="mpiexec -np ${NUM_PROCS} ${APP_NAME}"
echo "Executing: ${CMD}"
exec ${CMD}
