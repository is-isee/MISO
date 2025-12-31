#pragma once

#include <miso/backend.hpp>
#ifdef __CUDACC__
#include <miso/cuda_util.cuh>
#endif  // __CUDACC__

namespace miso {
namespace mhd {

/// @brief  Number of MHD fields
constexpr int n_fields = 9;

/// @brief Execution context for MHD
template <typename Backend = backend::Host> struct ExecContext {};

/// @brief Execution context for MHD on CPU
template <> struct ExecContext<backend::Host> {
  using memory_space = backend::Host;
  using Backend = backend::Host;
  mpi::Shape &mpi_shape;
};

#ifdef __CUDACC__
/// @brief Execution context for MHD on GPU
template <> struct ExecContext<backend::CUDA> {
  using memory_space = backend::CUDA;
  using Backend = backend::CUDA;
  mpi::Shape &mpi_shape;
  cuda::KernelShape3D &cu_shape;
};
#endif  // __CUDACC__

}  // namespace mhd
}  // namespace miso
