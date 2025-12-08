#!/usr/bin/env bash
# Description: Executes mpiexec with environment variables
# Usage: ./run.sh [mpiexec_options] <program>

# Exit on error / undefined variable
set -eu

# Load environment variables
set -a
source "$(dirname "$0")/openmpi_config.env"
set +a

# Run command
exec mpiexec "$@"
