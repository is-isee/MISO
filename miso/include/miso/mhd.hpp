#pragma once

#include "array3d.hpp"
#include "env.hpp"
#include "grid.hpp"
#include "mhd_checkpoint.hpp"
#include "mhd_fields.hpp"
#include "mhd_halo_exchange.hpp"
#include "mhd_integrator.hpp"
#include "time.hpp"

namespace miso {
namespace mhd {

template <typename Real, typename Backend> struct MHD {
  Grid<Real, Backend> grid;
  Fields<Real, Backend> qq;
  ExecContext<Backend> &exec_ctx;
  Integrator<Real, Backend> integrator;
  Checkpoint<Real> checkpoint;

  template <typename ExecContextType>
  MHD(Config &config, Grid<Real, backend::Host> &grid_h,
      ExecContextType &exec_ctx)
      : grid(grid_h), qq(grid_h), exec_ctx(exec_ctx),
        integrator(config, grid, exec_ctx), checkpoint(config, grid_h) {}

  template <typename InitialCondition, typename BoundaryCondition>
  void apply_initial_condition(const InitialCondition &ic,
                               const BoundaryCondition &bc) {
    Grid<Real, backend::Host> grid_h(grid);
    Fields<Real, backend::Host> qq_h(grid.i_total, grid.j_total, grid.k_total);
    ic.apply(qq_h.view(), grid_h.const_view());
    qq.copy_from(qq_h);
    integrator.apply_boundary_condition(bc, qq);
  }

  template <typename EOS> Real cfl(const EOS &eos) {
    return integrator.cfl(qq, eos);
  }

  template <typename EOS, typename BoundaryCondition, typename Source>
  void update(Real dt, const EOS &eos, const BoundaryCondition &bc,
              const Source &src) {
    integrator.update(dt, eos, bc, src, qq);
  }

  void save(const Time<Real> &time) { checkpoint.save(time, qq); };

  void load(const Time<Real> &time) { checkpoint.load(time, qq); };
};

}  // namespace mhd
}  // namespace miso
