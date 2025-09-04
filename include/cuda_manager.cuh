#pragma once
#include "grid_cpu.hpp"
#include "mpi_manager.hpp"
#include <cuda_runtime.h>

// clang-format off
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
// clang-format on

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      std::exit(code);
  }
};

template <typename Real> struct CudaManager {
  cudaStream_t stream_ro;
  cudaStream_t stream_vx;
  cudaStream_t stream_vy;
  cudaStream_t stream_vz;
  cudaStream_t stream_bx;
  cudaStream_t stream_by;
  cudaStream_t stream_bz;
  cudaStream_t stream_ei;
  cudaStream_t stream_ph;

  dim3 grid_dim, grid_dim_x_margin, grid_dim_y_margin, grid_dim_z_margin;
  dim3 block_dim;

  CudaManager(const Grid<Real> &grid) {

    CUDA_CHECK(cudaStreamCreate(&stream_ro));
    CUDA_CHECK(cudaStreamCreate(&stream_vx));
    CUDA_CHECK(cudaStreamCreate(&stream_vy));
    CUDA_CHECK(cudaStreamCreate(&stream_vz));
    CUDA_CHECK(cudaStreamCreate(&stream_bx));
    CUDA_CHECK(cudaStreamCreate(&stream_by));
    CUDA_CHECK(cudaStreamCreate(&stream_bz));
    CUDA_CHECK(cudaStreamCreate(&stream_ei));
    CUDA_CHECK(cudaStreamCreate(&stream_ph));

    block_dim = dim3(8, 8, 8);
    grid_dim = dim3((grid.i_total + block_dim.x - 1) / block_dim.x,
                    (grid.j_total + block_dim.y - 1) / block_dim.y,
                    (grid.k_total + block_dim.z - 1) / block_dim.z);

    grid_dim_x_margin = dim3((grid.i_margin + block_dim.x - 1) / block_dim.x,
                             (grid.j_total + block_dim.y - 1) / block_dim.y,
                             (grid.k_total + block_dim.z - 1) / block_dim.z);

    grid_dim_y_margin = dim3((grid.i_total + block_dim.x - 1) / block_dim.x,
                             (grid.j_margin + block_dim.y - 1) / block_dim.y,
                             (grid.k_total + block_dim.z - 1) / block_dim.z);

    grid_dim_z_margin = dim3((grid.i_total + block_dim.x - 1) / block_dim.x,
                             (grid.j_total + block_dim.y - 1) / block_dim.y,
                             (grid.k_margin + block_dim.z - 1) / block_dim.z);
  }
};
