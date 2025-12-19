#pragma once

#include <miso/cuda_utils.cuh>

namespace miso {

namespace mhd {

template <typename Real> struct TimeDevice {
  Real *dt_mins_d = nullptr;
  Real *dt_mins_h = nullptr;
  size_t shared_mem_size = 0;
  int n_blocks;

  TimeDevice(CudaKernelShape<Real> &cu_shape)
      : n_blocks(cu_shape.grid_dim.x * cu_shape.grid_dim.y *
                 cu_shape.grid_dim.z) {
    dt_mins_h = new Real[n_blocks];
    shared_mem_size = sizeof(Real) * cu_shape.block_dim.x * cu_shape.block_dim.y *
                      cu_shape.block_dim.z;
    CUDA_CHECK(cudaMalloc(&dt_mins_d, sizeof(Real) * n_blocks));
  }

  void copy_to_host() {
    CUDA_CHECK(cudaMemcpy(dt_mins_h, dt_mins_d, sizeof(Real) * n_blocks,
                          cudaMemcpyDeviceToHost));
  }

  void copy_to_device() {
    CUDA_CHECK(cudaMemcpy(dt_mins_d, dt_mins_h, sizeof(Real) * n_blocks,
                          cudaMemcpyHostToDevice));
  }
};

}  // namespace mhd

}  // namespace miso
