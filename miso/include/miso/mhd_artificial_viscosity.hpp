#pragma once

#include "mhd_artificial_viscosity_host.hpp"
#include "mhd_core.hpp"

#ifdef __CUDACC__
#include "mhd_artificial_viscosity_cuda.cuh"
#endif

namespace miso {
namespace mhd {

/// @brief Artificial viscosity of the backend.
/// @details A thin front that gives both implementations the same
/// constructor, so that backend-independent code (Integrator) can hold it as
/// ArtificialViscosity<Real, Backend>. The implementations themselves are
/// still separate (impl_host / impl_cuda); they are to be merged in the same
/// way as Integrator.
template <typename Real, typename Backend> struct ArtificialViscosity;

template <typename Real>
struct ArtificialViscosity<Real, backend::Host>
    : impl_host::ArtificialViscosity<Real> {
  ArtificialViscosity(Config &config, Grid<Real, backend::Host> &grid,
                      ExecContext<Real, backend::Host> &)
      : impl_host::ArtificialViscosity<Real>(config, grid) {}
};

#ifdef __CUDACC__
template <typename Real>
struct ArtificialViscosity<Real, backend::CUDA>
    : impl_cuda::ArtificialViscosity<Real> {
  ArtificialViscosity(Config &config, Grid<Real, backend::CUDA> &grid,
                      ExecContext<Real, backend::CUDA> &exec_ctx)
      : impl_cuda::ArtificialViscosity<Real>(config, grid, exec_ctx.cu_shape) {}
};
#endif  // __CUDACC__

}  // namespace mhd
}  // namespace miso
