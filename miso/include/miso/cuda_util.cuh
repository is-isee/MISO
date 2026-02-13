#pragma once

#include <cstdio>

#include <miso/cuda_compat.hpp>
#include <miso/env.hpp>

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

struct KernelShape3D {
  dim3 block_dim;
  dim3 grid_dim;
  dim3 grid_dim_x_margin, grid_dim_y_margin, grid_dim_z_margin;

  template <typename GridType>
  explicit KernelShape3D(const GridType &grid) noexcept {
    const auto div_up = [](int n, int d) { return (n + d - 1) / d; };

    block_dim = dim3(8, 8, 8);
    // clang-format off
    grid_dim = dim3(div_up(grid.i_total, block_dim.x),
                    div_up(grid.j_total, block_dim.y),
                    div_up(grid.k_total, block_dim.z));
    grid_dim_x_margin = dim3(div_up(grid.i_margin, block_dim.x),
                             div_up(grid.j_total, block_dim.y),
                             div_up(grid.k_total, block_dim.z));
    grid_dim_y_margin = dim3(div_up(grid.i_total, block_dim.x),
                             div_up(grid.j_margin, block_dim.y),
                             div_up(grid.k_total, block_dim.z));
    grid_dim_z_margin = dim3(div_up(grid.i_total, block_dim.x),
                             div_up(grid.j_total, block_dim.y),
                             div_up(grid.k_margin, block_dim.z));
    // clang-format on
  }
};

}  // namespace cuda
}  // namespace miso
