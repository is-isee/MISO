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
template <typename Backend = HostBackend> struct ExecContext;

/// @brief Execution context for MHD on CPU
struct ExecContext<HostBackend> {
  using memory_space = HostSpace;
  using backend = HostBackend;
  mpi::Shape &mpi_shape;
};

#ifdef USE_CUDA
/// @brief Execution context for MHD on GPU
struct ExecContext<CUDABackend> {
  using memory_space = CUDASpace;
  using backend = CUDABackend;
  mpi::Shape &mpi_shape;
  cuda::KernelShape3D &cu_shape;
};
#endif  // USE_CUDA

}  // namespace mhd
}  // namespace miso
