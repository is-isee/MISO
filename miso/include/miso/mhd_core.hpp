#pragma once

#include "backend.hpp"
#ifdef __CUDACC__
#include "cuda_util.cuh"
#endif  // __CUDACC__

namespace miso {
namespace mhd {

/// @brief  Number of MHD fields
constexpr int n_fields = 9;

/// @brief Execution context for MHD
template <typename Backend = backend::Host> struct ExecContext {};

/// @brief Execution context for MHD on CPU
template <> struct ExecContext<backend::Host> {
  mpi::Shape &mpi_shape;

  ExecContext(mpi::Shape &mpi_shape_, const Grid<Real, backend::Host> &)
      : mpi_shape(mpi_shape_) {}
};

#ifdef __CUDACC__
/// @brief Execution context for MHD on GPU
template <> struct ExecContext<backend::CUDA> {
  mpi::Shape &mpi_shape;
  cuda::KernelShape3D cu_shape;

  ExecContext(mpi::Shape &mpi_shape_, const Grid<Real, backend::Host> &grid)
      : mpi_shape(mpi_shape_), cu_shape(grid) {}
};
#endif  // __CUDACC__

}  // namespace mhd
}  // namespace miso
