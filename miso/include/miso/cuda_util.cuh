#pragma once

#include <cstdio>

#include "cuda_compat.hpp"
#include "env.hpp"

// clang-format off
/// @brief Macro to check CUDA errors
#define MISO_CUDA_CHECK(ans) \
  do { miso::cuda::check_error((ans), __FILE__, __LINE__); } while (0)
// clang-format on

namespace miso {
namespace cuda {

inline void check_error(cudaError_t code, const char *file, int line,
                        bool abort = true) {
  if (code != cudaSuccess) {
    const char *errstr = cudaGetErrorString(code);
    std::fprintf(stderr, "CUDA Error: %s %s %d\n", errstr, file, line);
    std::fflush(stderr);
    if (abort) {
      int is_mpi_initialized = false;
      MPI_Initialized(&is_mpi_initialized);
      if (is_mpi_initialized) {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      } else {
        std::abort();
      }
    }
  }
}

}  // namespace cuda
}  // namespace miso
