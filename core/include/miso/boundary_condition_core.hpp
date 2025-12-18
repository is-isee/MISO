#pragma once

#include <miso/boundary_condition_core_shared.hpp>

#ifdef USE_CUDA
#include <miso/boundary_condition_core_gpu.cuh>
#endif

#include <miso/boundary_condition_core_cpu.hpp>
