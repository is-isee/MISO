#pragma once

#include <miso/array3d.hpp>
#include <miso/env.hpp>
#include <miso/eos.hpp>
#include <miso/grid.hpp>
#include <miso/mhd_checkpoint.hpp>
#include <miso/mhd_fields.hpp>
#include <miso/mhd_halo_exchange.hpp>
#include <miso/mhd_integrator.hpp>
#include <miso/time.hpp>

namespace miso {
namespace mhd {

template <typename Real, typename EOS, typename Backend> struct MHD {
  Grid<Real, Backend> grid;
  Fields<Real, Backend> qq;
  ExecContext<Backend> &exec_ctx;
  Integrator<Real, EOS, Backend> integrator;
  Checkpoint<Real> checkpoint;

  template <typename ExecContextType>
  MHD(Config &config, Grid<Real, backend::Host> &grid_h,
      ExecContextType &exec_ctx, EOS &eos)
      : grid(grid_h), qq(grid_h), exec_ctx(exec_ctx),
        integrator(config, grid, exec_ctx, eos), checkpoint(config, grid_h) {}

  template <typename InitialCondition, typename BoundaryCondition>
  void apply_initial_condition(const InitialCondition &ic,
                               const BoundaryCondition &bc,
                               Grid<Real, backend::Host> &grid_h) {
    /// @todo: Should be initialized by the specified backend.
    Fields<Real, backend::Host> qq_i(grid_h);
    ic.apply(qq_i.view(), grid_h.const_view(), integrator.eos);
    qq.copy_from(qq_i);
    integrator.apply_boundary_condition(bc, qq);
  }

  Real cfl() const { return integrator.cfl(qq); }

  template <typename BoundaryCondition, typename Source>
  void update(Real dt, const BoundaryCondition &bc, const Source &src) {
    integrator.update(dt, bc, src, qq);
  }

  void save(const Time<Real> &time) { checkpoint.save(time, qq); };

  void load(const Time<Real> &time) { checkpoint.load(time, qq); };
};

}  // namespace mhd
}  // namespace miso
