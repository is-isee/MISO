#!/usr/bin/env bash
# Description: Build and run the MISO application

# Exit on error / undefined variable
set -eu

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
CONFIG_PATH="${THIS_DIR}/config.yaml"
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Error: Config file not found at ${CONFIG_PATH}"
    exit 1
fi

# Match the checked-in MPI topology from config.yaml.
NUM_PROCS=$(awk '
    $1 == "x_procs:" { x = $2 }
    $1 == "y_procs:" { y = $2 }
    $1 == "z_procs:" { z = $2 }
    END {
        if (x == "" || y == "" || z == "") {
            exit 1
        }
        print x * y * z
    }
' "${CONFIG_PATH}") || {
    echo "Error: Failed to read x_procs/y_procs/z_procs from ${CONFIG_PATH}"
    exit 1
}

# Run command
set -x
mpiexec -np ${NUM_PROCS} "${APP_NAME}" --config="${CONFIG_PATH}"
