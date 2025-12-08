#!/usr/bin/env bash

set -eu

# Default values
USE_CUDA=false
NUM_PROCS=1

# Analyze command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --cuda)
      USE_CUDA=true
      shift
      ;;
    -np)
      NUM_PROCS="$2"
      shift 2
      ;;
    *)
      echo "Error: Unknown argument '$1'" >&2
      echo "Usage: $0 [--cuda] [--np <num>]" >&2
      exit 1
      ;;
  esac
done

# Define application name
APP_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_NAME=$(basename "${APP_DIR}")
APP_NAME="${APP_DIR}/build/${PROJECT_NAME}"

# MPI options for CUDA MPI
CUDA_MPI_OPTS="--bind-to none"
CUDA_MPI_OPTS+=" --mca pml ob1"
CUDA_MPI_OPTS+=" --mca btl tcp,self,vader"
CUDA_MPI_OPTS+=" --mca coll ^hcoll"
CUDA_MPI_OPTS+=" --mca osc ^ucx"

# Run the application
if [ "$USE_CUDA" = true ]; then
  CMD="mpiexec -np ${NUM_PROCS} ${CUDA_MPI_OPTS} ${APP_NAME}"
else
  CMD="mpiexec -np ${NUM_PROCS} ${APP_NAME}"
fi
echo $CMD
exec $CMD
