#pragma once

#include "backend.hpp"
#include "grid.hpp"
#include "mpi_util.hpp"

namespace miso {
namespace mhd {

/// @brief  Number of MHD fields
constexpr int n_fields = 9;

/// @brief Execution context for MHD
/// @details The same for all backends. Launch configurations of the CUDA
/// backend are chosen by for_each (execution.hpp).
template <typename Real, typename Backend> struct ExecContext {
  mpi::Shape &mpi_shape;

  ExecContext(mpi::Shape &mpi_shape_, const Grid<Real, backend::Host> &)
      : mpi_shape(mpi_shape_) {}
};

}  // namespace mhd
}  // namespace miso
