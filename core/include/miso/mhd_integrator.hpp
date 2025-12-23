#pragma once

#ifdef USE_CUDA
#include <miso/mhd_integrator_gpu.cuh>
#else
#include <miso/mhd_integrator_cpu.hpp>
#endif
