#pragma once

#ifdef USE_CUDA

#include <cuda_runtime.h>

#else  // USE_CUDA

#ifndef __device__
#define __device__
#endif

#ifndef __host__
#define __host__
#endif

// Do not define __global__:
// CUDA kernel semantics and launch syntax are not portable.

#endif  // USE_CUDA
