#pragma once

#include <miso/cuda_compat.cuh>

namespace miso {

template <typename Real> struct Grid;

namespace mhd {

template <typename Real> struct MHDCudaManager {
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

  MHDCudaManager(const Grid<Real> &grid) {

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

}  // namespace mhd

}  // namespace miso
