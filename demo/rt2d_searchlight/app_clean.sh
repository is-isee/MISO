#!/usr/bin/env bash
# Description: Clean output data of the MISO application

set -eu

THIS_DIR=$(cd "$(dirname "$0")" && pwd)

set -x
rm -rf "${THIS_DIR}"/build "${THIS_DIR}"/figs "${THIS_DIR}"/data
