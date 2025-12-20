/// @brief CUDA compatibility macros and functions
#pragma once

#if defined(__CUDACC__)
#include <cstdio>
#include <cuda_runtime.h>

#define HOST __host__
#define DEVICE __device__
#define HOST_DEVICE __host__ __device__
#define GLOBAL __global__

// clang-format off
#define CUDA_CHECK(ans) { miso::gpuAssert((ans), __FILE__, __LINE__); }
// clang-format on

namespace miso {

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    std::fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file,
                 line);
    if (abort)
      std::exit(code);
  }
};

}  // namespace miso

#else
#define HOST
#define DEVICE
#define HOST_DEVICE
#define GLOBAL
#define CUDA_CHECK(ans)
#endif  // defined(__CUDACC__)
