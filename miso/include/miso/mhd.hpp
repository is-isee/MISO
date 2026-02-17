#pragma once

#include "array3d.hpp"
#include "env.hpp"
#include "eos.hpp"
#include "grid.hpp"
#include "mhd_checkpoint.hpp"
#include "mhd_fields.hpp"
#include "mhd_halo_exchange.hpp"
#include "mhd_integrator.hpp"
#include "time.hpp"

namespace miso {
namespace mhd {

template <typename Real, typename EOS, typename Backend> struct MHD {
  Grid<Real, backend::Host> &grid_host;  // for initial condition
  Grid<Real, Backend> grid;
  Fields<Real, Backend> qq;
  ExecContext<Backend> &exec_ctx;
  Integrator<Real, EOS, Backend> integrator;
  Checkpoint<Real> checkpoint;

  template <typename ExecContextType>
  MHD(Config &config, Grid<Real, backend::Host> &grid_h,
      ExecContextType &exec_ctx, EOS &eos)
      : grid_host(grid_h), grid(grid_h), qq(grid_h), exec_ctx(exec_ctx),
        integrator(config, grid, exec_ctx, eos), checkpoint(config, grid_h) {}

  template <typename InitialCondition, typename BoundaryCondition>
  void apply_initial_condition(const InitialCondition &ic,
                               const BoundaryCondition &bc) {
    Fields<Real, backend::Host> qq_h(grid_host);
    ic.apply(qq_h.view(), grid_host.const_view(), integrator.eos);
    qq.copy_from(qq_h);
    integrator.apply_boundary_condition(bc, qq);
  }

  Real cfl() { return integrator.cfl(qq); }

  template <typename BoundaryCondition, typename Source>
  void update(Real dt, const BoundaryCondition &bc, const Source &src) {
    integrator.update(dt, bc, src, qq);
  }

  void save(const Time<Real> &time) { checkpoint.save(time, qq); };

  void load(const Time<Real> &time) { checkpoint.load(time, qq); };
};

}  // namespace mhd
}  // namespace miso
