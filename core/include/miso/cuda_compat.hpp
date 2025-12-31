#pragma once

#ifdef __CUDACC__

#include <cuda_runtime.h>

#else  // __CUDACC__

#ifndef __device__
#define __device__
#endif

#ifndef __host__
#define __host__
#endif

// Do not define __global__:
// CUDA kernel semantics and launch syntax are not portable.

#endif  // __CUDACC__
