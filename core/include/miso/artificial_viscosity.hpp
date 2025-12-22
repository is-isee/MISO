#pragma once

#ifdef __CUDACC__
#include <miso/artificial_viscosity_gpu.cuh>
#else
#include <miso/artificial_viscosity_cpu.hpp>
#endif
