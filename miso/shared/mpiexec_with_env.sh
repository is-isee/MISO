#!/usr/bin/env bash
# Description: Executes mpiexec with environment variables
# Usage: ./run.sh [mpiexec_options] <program>

# Exit on error / undefined variable
set -eu

# Load environment variables
set -a
source "$(cd "$(dirname "$0")" && pwd)/openmpi_config.env"
set +a

# Run command
exec mpiexec "$@"
