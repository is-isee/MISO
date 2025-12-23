#pragma once

#ifdef USE_CUDA
#include <miso/mhd_artificial_viscosity_gpu.cuh>
#else
#include <miso/mhd_artificial_viscosity_cpu.hpp>
#endif
