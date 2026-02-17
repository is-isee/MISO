#pragma once

#include "mhd_artificial_viscosity_host.hpp"

#ifdef __CUDACC__
#include "mhd_artificial_viscosity_cuda.cuh"
#endif
