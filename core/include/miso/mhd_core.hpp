#pragma once

#include <miso/policy.hpp>
#ifdef USE_CUDA
#include <miso/cuda_util.cuh>
#endif  // USE_CUDA

namespace miso {
namespace mhd {

/// @brief  Number of MHD fields
constexpr int n_fields = 9;

/// @brief Execution context for MHD
template <typename Space = HostSpace> struct ExecContext {};

/// @brief Execution context for MHD on CPU
template <> struct ExecContext<HostSpace> {
  using memory_space = HostSpace;
  using Space = HostSpace;
  mpi::Shape &mpi_shape;
};

#ifdef USE_CUDA
/// @brief Execution context for MHD on GPU
template <> struct ExecContext<CUDASpace> {
  using memory_space = CUDASpace;
  using Space = CUDASpace;
  mpi::Shape &mpi_shape;
  cuda::KernelShape3D &cu_shape;
};
#endif  // USE_CUDA

}  // namespace mhd
}  // namespace miso
