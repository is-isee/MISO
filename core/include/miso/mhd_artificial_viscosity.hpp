#pragma once

#ifdef __CUDACC__
#include <miso/mhd_artificial_viscosity_gpu.cuh>
#else
#include <miso/mhd_artificial_viscosity_cpu.hpp>
#endif
