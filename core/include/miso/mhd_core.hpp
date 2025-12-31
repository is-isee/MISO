#pragma once

#include <miso/backend.hpp>
#ifdef USE_CUDA
#include <miso/cuda_util.cuh>
#endif  // USE_CUDA

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

#ifdef USE_CUDA
/// @brief Execution context for MHD on GPU
template <> struct ExecContext<backend::CUDA> {
  using memory_space = backend::CUDA;
  using Backend = backend::CUDA;
  mpi::Shape &mpi_shape;
  cuda::KernelShape3D &cu_shape;
};
#endif  // USE_CUDA

}  // namespace mhd
}  // namespace miso
