#pragma once

#include <miso/boundary_condition_core.hpp>
#include <miso/boundary_condition_cpu.hpp>

#ifdef __CUDACC__
#include <miso/boundary_condition_gpu.cuh>
#endif
