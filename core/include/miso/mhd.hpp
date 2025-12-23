#pragma once

// #include <miso/mhd_core.hpp>

#include <miso/mhd_cpu.hpp>

#ifdef USE_CUDA
#include <miso/mhd_gpu.cuh>
// #else
// #include <miso/mhd_cpu.hpp>
#endif
