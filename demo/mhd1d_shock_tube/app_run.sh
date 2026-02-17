#!/usr/bin/env bash
# Description: Build and run the MISO application

# Exit on error / undefined variable
set -eu

# Number of processes
NUM_PROCS=1

# Define directory of this script
THIS_DIR=$(cd "$(dirname "$0")" && pwd)

# Define application name
APP_NAME="${THIS_DIR}/build/$(basename "${THIS_DIR}")"
if [[ ! -f "${APP_NAME}" ]]; then
    echo "Error: Application binary not found at ${APP_NAME}"
    exit 1
fi

# Load environment variables
set -a
source "${THIS_DIR}"/../shared/openmpi_config.env
set +a

# Config file path
CONFIG_PATH_X="${THIS_DIR}/config/config_x.yaml"
if [[ ! -f "${CONFIG_PATH_X}" ]]; then
    echo "Error: Config file not found at ${CONFIG_PATH_X}"
    exit 1
fi
CONFIG_PATH_Y="${THIS_DIR}/config/config_y.yaml"
if [[ ! -f "${CONFIG_PATH_Y}" ]]; then
    echo "Error: Config file not found at ${CONFIG_PATH_Y}"
    exit 1
fi
CONFIG_PATH_Z="${THIS_DIR}/config/config_z.yaml"
if [[ ! -f "${CONFIG_PATH_Z}" ]]; then
    echo "Error: Config file not found at ${CONFIG_PATH_Z}"
    exit 1
fi

# Run command
set -x
mpiexec -np ${NUM_PROCS} "${APP_NAME}" --config="${CONFIG_PATH_X}"
mpiexec -np ${NUM_PROCS} "${APP_NAME}" --config="${CONFIG_PATH_Y}"
mpiexec -np ${NUM_PROCS} "${APP_NAME}" --config="${CONFIG_PATH_Z}"
