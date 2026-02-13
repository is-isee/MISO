#include "common.hpp"

struct TimeStep {
  Real cfl_number;
  Real rra;

  TimeStep(Config &config, Grid<Real, Backend> &grid) {
    cfl_number = config["mhd"]["cfl_number"].as<Real>();
    rra = config["magnetosphere"]["radius"].as<Real>();
  }

  // CFL condition with the inner boundary region masked.
  template <typename EOS>
  Real cfl(const mhd::Fields<Real, Backend> &qq, const Grid<Real, Backend> &grid,
           EOS &eos) {
    const Real slow_speed = 1.e-10;
    const Real dt_max = 1.e10;

    auto qq_v = qq.const_view();
    auto grid_v = grid.const_view();

    Range3D range{{grid.i_margin, grid.i_total - grid.i_margin},
                  {grid.j_margin, grid.j_total - grid.j_margin},
                  {grid.k_margin, grid.k_total - grid.k_margin}};
    const auto f = MISO_LAMBDA(int i, int j, int k) {
      Real cs = util::sqrt(eos.gm * (eos.gm - 1.0) * qq_v.ei(i, j, k));
      Real vv = util::sqrt(qq_v.vx(i, j, k) * qq_v.vx(i, j, k) +
                           qq_v.vy(i, j, k) * qq_v.vy(i, j, k) +
                           qq_v.vz(i, j, k) * qq_v.vz(i, j, k));
      Real ca = util::sqrt((qq_v.bx(i, j, k) * qq_v.bx(i, j, k) +
                            qq_v.by(i, j, k) * qq_v.by(i, j, k) +
                            qq_v.bz(i, j, k) * qq_v.bz(i, j, k)) /
                           qq_v.ro(i, j, k) * pii4<Real>);
      Real total_vel = (cs + vv + ca);

      Real rr = util::sqrt(util::pow2(grid_v.x[i]) + util::pow2(grid_v.y[j]) +
                           util::pow2(grid_v.z[k]));
      Real mask = rr > rra ? 1.0 : 0.0;
      Real masked_vel = total_vel * mask + slow_speed * (1.0 - mask);

      Real dxyz = util::min3(grid_v.dx[i], grid_v.dy[j], grid_v.dz[k]);
      return cfl_number * dxyz / masked_vel;
    };
    const auto op = MISO_LAMBDA(Real a, Real b) { return util::min2(a, b); };
    const auto dt = reduce(Backend{}, range, dt_max, f, op);

    Real dt_g;
    MPI_Allreduce(&dt, &dt_g, 1, mpi::data_type<Real>(), MPI_MIN, mpi::comm());
    return dt_g;
  }
};
