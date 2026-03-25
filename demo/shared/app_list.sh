#!/usr/bin/env bash
# Description: List of demo applications

# Exit on error / undefined variable
set -eu

# List of demo applications
APP_LIST=(
  "hd1d_shock_tube"
  "hd2d_kelvin_helmholtz"
  "hd2d_rayleigh_taylor"
  "mhd1d_shock_tube"
  "mhd2d_vortex"
  "mhd3d_magnetosphere"
  "rt2d_searchlight"
)
