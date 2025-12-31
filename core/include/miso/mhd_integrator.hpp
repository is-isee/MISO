#pragma once

#ifdef __CUDACC__
#include <miso/mhd_integrator_gpu.cuh>
#else
#include <miso/mhd_integrator_cpu.hpp>
#endif
