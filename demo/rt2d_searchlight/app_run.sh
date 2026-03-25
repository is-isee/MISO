#!/usr/bin/env bash
# Description: Build and run the MISO application

set -eu

NUM_PROCS=1

THIS_DIR=$(cd "$(dirname "$0")" && pwd)

APP_NAME="${THIS_DIR}/build/$(basename "${THIS_DIR}")"
if [[ ! -f "${APP_NAME}" ]]; then
    echo "Error: Application binary not found at ${APP_NAME}"
    exit 1
fi

set -a
source "${THIS_DIR}"/../shared/openmpi_config.env
set +a

CONFIG_PATH="${THIS_DIR}/config.yaml"
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Error: Config file not found at ${CONFIG_PATH}"
    exit 1
fi

set -x
mpiexec -np ${NUM_PROCS} "${APP_NAME}" --config="${CONFIG_PATH}"
