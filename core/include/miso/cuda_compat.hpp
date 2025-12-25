#pragma once

#ifdef USE_CUDA
#include <cuda_runtime.h>
#define HOST __host__
#define DEVICE __device__
#define HOST_DEVICE __host__ __device__
#define GLOBAL __global__
#else
#define HOST
#define DEVICE
#define HOST_DEVICE
#define GLOBAL
#endif
