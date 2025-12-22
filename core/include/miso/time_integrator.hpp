#pragma once

#ifdef __CUDACC__
#include <miso/time_integrator_gpu.cuh>
#else
#include <miso/time_integrator_cpu.hpp>
#endif
