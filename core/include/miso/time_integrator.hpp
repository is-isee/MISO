#pragma once

#ifdef USE_CUDA
#include <miso/time_integrator_gpu.cuh>
#else
#include <miso/time_integrator_cpu.hpp>
#endif
