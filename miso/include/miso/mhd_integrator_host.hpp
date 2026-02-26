#pragma once

#include "array3d.hpp"
#include "constants.hpp"
#include "env.hpp"
#include "mhd_artificial_viscosity_host.hpp"
#include "mhd_fields.hpp"

namespace miso {
namespace mhd {

/// @brief Class for time integration of MHD equations
template <typename Real> struct Integrator<Real, backend::Host> {
  /// @brief Spatial grid
  Grid<Real, backend::Host> &grid;
  /// @brief Workspace
  Fields<Real, backend::Host> qq_argm, qq_rslt;

  /// @brief Halo exchanger
  HaloExchanger<Real, backend::Host> halo_exchanger;
  /// @brief Artificial viscosity for MHD equations
  impl_host::ArtificialViscosity<Real> artdiff;

  /// @brief gas pressure
  Array3D<Real, backend::Host> pr;
  /// @brief magnetic field strength bx*bx + by*by + bz*bz
  Array3D<Real, backend::Host> bb;
  /// @brief enthalpy + 2*magnetic energy + kinetic energy
  Array3D<Real, backend::Host> ht;
  /// @brief inner product of velocity and magnetic field vx*bx + vy*by + vz*bz
  Array3D<Real, backend::Host> vb;
  /// @brief speed of sound
  Array3D<Real, backend::Host> cs;

  /// @brief CFL number
  Real cfl_number;
  /// @brief propagation speed fo divergence B
  Real ch_divb;
  /// @brief square of ch_divb;
  Real ch_divb_square;
  /// @brief damping time scape for divergence B
  Real tau_divb;

  /// @brief Constructor
  Integrator(Config &config, Grid<Real, backend::Host> &grid,
             ExecContext<Real, backend::Host> &exec_ctx)
      : grid(grid), qq_argm(grid), qq_rslt(grid), halo_exchanger(grid, exec_ctx),
        artdiff(config, grid), pr(grid.i_total, grid.j_total, grid.k_total),
        bb(grid.i_total, grid.j_total, grid.k_total),
        ht(grid.i_total, grid.j_total, grid.k_total),
        vb(grid.i_total, grid.j_total, grid.k_total),
        cs(grid.i_total, grid.j_total, grid.k_total) {
    cfl_number = config["mhd"]["cfl_number"].as<Real>();
  }

  /// @brief Update MHD equations using 4th order space-centered scheme
  template <typename EOS, typename Source>
  void update_sc4(const Real dt, const EOS &eos, const Source &src,
                  const Fields<Real> &qq_orgn, const Fields<Real> &qq_argm,
                  Fields<Real> &qq_rslt) {
    // gas pressure
    eos.gas_pressure(backend::Host{}, qq_argm.const_view(), pr.view());

    for (int i = 0; i < grid.i_total; ++i) {
      for (int j = 0; j < grid.j_total; ++j) {
        for (int k = 0; k < grid.k_total; ++k) {
          // clang-format off
          // squared magnetic strength
          bb(i, j, k) = qq_argm.bx(i,j,k)*qq_argm.bx(i,j,k)
                      + qq_argm.by(i,j,k)*qq_argm.by(i,j,k)
                      + qq_argm.bz(i,j,k)*qq_argm.bz(i,j,k);
          // enthalpy + 2*magnetic energy + kinetic energy
          ht(i, j, k) =
              + qq_argm.ro(i,j,k)*qq_argm.ei(i,j,k) + pr(i,j,k)
              + bb(i,j,k)*pii4<Real>
              + 0.5*qq_argm.ro(i,j,k)*(
                  + qq_argm.vx(i,j,k)*qq_argm.vx(i,j,k)
                  + qq_argm.vy(i,j,k)*qq_argm.vy(i,j,k)
                  + qq_argm.vz(i,j,k)*qq_argm.vz(i,j,k)
              );
          // v dot b
          vb(i, j, k) =
              + qq_argm.vx(i,j,k)*qq_argm.bx(i,j,k)
              + qq_argm.vy(i,j,k)*qq_argm.by(i,j,k)
              + qq_argm.vz(i,j,k)*qq_argm.bz(i,j,k);
          // clang-format on
        }
      }
    }

    // Generate const view
    const auto c_pr = pr.const_view();
    const auto c_bb = bb.const_view();
    const auto c_ht = ht.const_view();
    const auto c_vb = vb.const_view();
    const auto c_qq = qq_argm.const_view();
    const auto c_ro = c_qq.ro;
    const auto c_vx = c_qq.vx;
    const auto c_vy = c_qq.vy;
    const auto c_vz = c_qq.vz;
    const auto c_bx = c_qq.bx;
    const auto c_by = c_qq.by;
    const auto c_bz = c_qq.bz;
    const auto c_ph = c_qq.ph;

    for (int i = grid.i_margin; i < grid.i_total - grid.i_margin; ++i) {
      const Real dxi = grid.dxi[i];
      for (int j = grid.j_margin; j < grid.j_total - grid.j_margin; ++j) {
        const Real dyi = grid.dyi[j];
        for (int k = grid.k_margin; k < grid.k_total - grid.k_margin; ++k) {
          const Real dzi = grid.dzi[k];

          // clang-format off
          // equation of continuity
          qq_rslt.ro(i, j, k) = qq_orgn.ro(i, j, k) + dt * (
          -space_centered_4th(c_ro, c_vx, dxi, i, j, k, grid.is, 0, 0)
          -space_centered_4th(c_ro, c_vy, dyi, i, j, k, 0, grid.js, 0)
          -space_centered_4th(c_ro, c_vz, dzi, i, j, k, 0, 0, grid.ks)
          );

          // x equation of motion
          qq_rslt.vx(i, j, k) = (
              qq_orgn.ro(i, j, k) * qq_orgn.vx(i, j, k) + dt * (
              -space_centered_4th(c_ro, c_vx, c_vx, dxi, i, j, k, grid.is, 0, 0)
              -space_centered_4th(c_ro, c_vx, c_vy, dyi, i, j, k, 0, grid.js, 0)
              -space_centered_4th(c_ro, c_vx, c_vz, dzi, i, j, k, 0, 0, grid.ks)
              -space_centered_4th(c_pr, dxi, i, j, k, grid.is, 0, 0)
              -space_centered_4th(c_bb, dxi, i, j, k, grid.is, 0, 0)*pii8<Real>
              +pii4<Real>*(
                  +space_centered_4th(c_bx, c_bx, dxi, i, j, k, grid.is, 0, 0)
                  +space_centered_4th(c_bx, c_by, dyi, i, j, k, 0, grid.js, 0)
                  +space_centered_4th(c_bx, c_bz, dzi, i, j, k, 0, 0, grid.ks)
                  )
              +src.vx(c_qq, i, j, k)
              )
          )/qq_rslt.ro(i, j, k);

          // y equation of motion
          qq_rslt.vy(i, j, k) = (
              qq_orgn.ro(i, j, k) * qq_orgn.vy(i, j, k) + dt * (
              -space_centered_4th(c_ro, c_vy, c_vx, dxi, i, j, k, grid.is, 0, 0)
              -space_centered_4th(c_ro, c_vy, c_vy, dyi, i, j, k, 0, grid.js, 0)
              -space_centered_4th(c_ro, c_vy, c_vz, dzi, i, j, k, 0, 0, grid.ks)
              -space_centered_4th(c_pr, dyi, i, j, k, 0, grid.js, 0)
              -space_centered_4th(c_bb, dyi, i, j, k, 0, grid.js, 0)*pii8<Real>
              +pii4<Real>*(
                  +space_centered_4th(c_by, c_bx, dxi, i, j, k, grid.is, 0, 0)
                  +space_centered_4th(c_by, c_by, dyi, i, j, k, 0, grid.js, 0)
                  +space_centered_4th(c_by, c_bz, dzi, i, j, k, 0, 0, grid.ks)
                  )
              +src.vy(c_qq, i, j, k)
              )
          )/qq_rslt.ro(i, j, k);

          // z equation of motion
          qq_rslt.vz(i, j, k) = (
              qq_orgn.ro(i, j, k) * qq_orgn.vz(i, j, k) + dt * (
              -space_centered_4th(c_ro, c_vz, c_vx, dxi, i, j, k, grid.is, 0, 0)
              -space_centered_4th(c_ro, c_vz, c_vy, dyi, i, j, k, 0, grid.js, 0)
              -space_centered_4th(c_ro, c_vz, c_vz, dzi, i, j, k, 0, 0, grid.ks)
              -space_centered_4th(c_pr, dzi, i, j, k, 0, 0, grid.ks)
              -space_centered_4th(c_bb, dzi, i, j, k, 0, 0, grid.ks)*pii8<Real>
              +pii4<Real>*(
                  +space_centered_4th(c_bz, c_bx, dxi, i, j, k, grid.is, 0, 0)
                  +space_centered_4th(c_bz, c_by, dyi, i, j, k, 0, grid.js, 0)
                  +space_centered_4th(c_bz, c_bz, dzi, i, j, k, 0, 0, grid.ks)
                  )
              +src.vz(c_qq, i, j, k)
              )
          )/qq_rslt.ro(i, j, k);

          // x magnetic induction
          qq_rslt.bx(i, j, k) = qq_orgn.bx(i, j, k) + dt * (
              -space_centered_4th(c_vy, c_bx, dyi, i, j, k, 0, grid.js, 0)
              -space_centered_4th(c_vz, c_bx, dzi, i, j, k, 0, 0, grid.ks)
              +space_centered_4th(c_vx, c_by, dyi, i, j, k, 0, grid.js, 0)
              +space_centered_4th(c_vx, c_bz, dzi, i, j, k, 0, 0, grid.ks)
              -space_centered_4th(c_ph, dxi, i, j, k, grid.is, 0, 0)
          );

          // y magnetic induction
          qq_rslt.by(i, j, k) = qq_orgn.by(i, j, k) + dt * (
              -space_centered_4th(c_vx, c_by, dxi, i, j, k, grid.is, 0, 0)
              -space_centered_4th(c_vz, c_by, dzi, i, j, k, 0, 0, grid.ks)
              +space_centered_4th(c_vy, c_bx, dxi, i, j, k, grid.is, 0, 0)
              +space_centered_4th(c_vy, c_bz, dzi, i, j, k, 0, 0, grid.ks)
              -space_centered_4th(c_ph, dyi, i, j, k, 0, grid.js, 0)
          );

          // z magnetic induction
          qq_rslt.bz(i, j, k) = qq_orgn.bz(i, j, k) + dt * (
              -space_centered_4th(c_vx, c_bz, dxi, i, j, k, grid.is, 0, 0)
              -space_centered_4th(c_vy, c_bz, dyi, i, j, k, 0, grid.js, 0)
              +space_centered_4th(c_vz, c_bx, dxi, i, j, k, grid.is, 0, 0)
              +space_centered_4th(c_vz, c_by, dyi, i, j, k, 0, grid.js, 0)
              -space_centered_4th(c_ph, dzi, i, j, k, 0, 0, grid.ks)
          );

          // div B factor
          qq_rslt.ph(i, j, k) = (
              qq_orgn.ph(i, j, k) + dt * ch_divb_square*(
              -space_centered_4th(c_bx, dxi, i, j, k, grid.is, 0, 0)
              -space_centered_4th(c_by, dyi, i, j, k, 0, grid.js, 0)
              -space_centered_4th(c_bz, dzi, i, j, k, 0, 0, grid.ks)
              )
          )*util::exp(-dt/tau_divb);

          // Et: total energy per unit volume
          // ei: is the internal energy per unit mass
          const Real Et =
              + qq_orgn.ro(i,j,k)*qq_orgn.ei(i,j,k)
              + 0.5*qq_orgn.ro(i, j, k)*(
                  + qq_orgn.vx(i, j, k)*qq_orgn.vx(i, j, k)
                  + qq_orgn.vy(i, j, k)*qq_orgn.vy(i, j, k)
                  + qq_orgn.vz(i, j, k)*qq_orgn.vz(i, j, k)
              )
              + pii8<Real>*(
                  + qq_orgn.bx(i,j,k)*qq_orgn.bx(i,j,k)
                  + qq_orgn.by(i,j,k)*qq_orgn.by(i,j,k)
                  + qq_orgn.bz(i,j,k)*qq_orgn.bz(i,j,k)
              );

          qq_rslt.ei(i, j, k) = ( Et + dt * (
              -space_centered_4th(c_ht, c_vx, dxi, i, j, k, grid.is, 0, 0)
              -space_centered_4th(c_ht, c_vy, dyi, i, j, k, 0, grid.js, 0)
              -space_centered_4th(c_ht, c_vz, dzi, i, j, k, 0, 0, grid.ks)
              + pii4<Real>* (
                  +space_centered_4th(c_vb, c_bx, dxi, i, j, k, grid.is, 0, 0)
                  +space_centered_4th(c_vb, c_by, dyi, i, j, k, 0, grid.js, 0)
                  +space_centered_4th(c_vb, c_bz, dzi, i, j, k, 0, 0, grid.ks)
                  )
              )
              -0.5*qq_rslt.ro(i, j, k)*(
                  + qq_rslt.vx(i, j, k)*qq_rslt.vx(i, j, k)
                  + qq_rslt.vy(i, j, k)*qq_rslt.vy(i, j, k)
                  + qq_rslt.vz(i, j, k)*qq_rslt.vz(i, j, k) )
              - pii8<Real>*(
                  + qq_rslt.bx(i, j, k)*qq_rslt.bx(i, j, k)
                  + qq_rslt.by(i, j, k)*qq_rslt.by(i, j, k)
                  + qq_rslt.bz(i, j, k)*qq_rslt.bz(i, j, k) )
              + src.ei(c_qq, i, j, k)
          )/qq_rslt.ro(i, j, k);
          // clang-format on
        }
      }
    }
  }

  /// @brief Apply boundary condition and halo exchange
  template <typename BoundaryCondition>
  void apply_boundary_condition(const BoundaryCondition &bc,
                                Fields<Real, backend::Host> &qq) {
    bc.apply(qq.view(), grid.const_view());
    halo_exchanger.apply(qq);
  }

  /// @brief Runge-Kutta 4th order time integration step
  template <typename EOS, typename BoundaryCondition, typename Source>
  void runge_kutta_4step(const Real dt, const EOS &eos,
                         const BoundaryCondition &bc, const Source &src,
                         Fields<Real, backend::Host> &qq) {
    update_sc4(dt / 4.0, eos, src, qq, qq, qq_rslt);
    qq_argm.copy_from(qq_rslt);
    apply_boundary_condition(bc, qq_argm);

    update_sc4(dt / 3.0, eos, src, qq, qq_argm, qq_rslt);
    qq_argm.copy_from(qq_rslt);
    apply_boundary_condition(bc, qq_argm);

    update_sc4(dt / 2.0, eos, src, qq, qq_argm, qq_rslt);
    qq_argm.copy_from(qq_rslt);
    apply_boundary_condition(bc, qq_argm);

    update_sc4(dt, eos, src, qq, qq_argm, qq_rslt);
    qq.copy_from(qq_rslt);
    apply_boundary_condition(bc, qq);
  }

  /// @brief Apply artificial viscosity
  template <typename EOS, typename BoundaryCondition>
  void apply_artificial_viscosity(const Real dt, const EOS &eos,
                                  const BoundaryCondition &bc,
                                  Fields<Real, backend::Host> &qq) {
    artdiff.characteristic_velocity_eval(qq, eos);

    // x direction
    artdiff.update(qq, qq_rslt, Direction::X, dt);
    qq.copy_from(qq_rslt);
    apply_boundary_condition(bc, qq);

    // y direction
    artdiff.update(qq, qq_rslt, Direction::Y, dt);
    qq.copy_from(qq_rslt);
    apply_boundary_condition(bc, qq);

    // z direction
    artdiff.update(qq, qq_rslt, Direction::Z, dt);
    qq.copy_from(qq_rslt);
    apply_boundary_condition(bc, qq);
  }

  /// @brief Calculate time spacing based on CFL condition
  template <typename EOS>
  Real cfl(const Fields<Real, backend::Host> &qq, const EOS &eos) {
    Real dt = 1.e10;
    eos.sound_speed(backend::Host{}, qq.const_view(), cs.view());
    for (int i = grid.i_margin; i < grid.i_total - grid.i_margin; ++i) {
      for (int j = grid.j_margin; j < grid.j_total - grid.j_margin; ++j) {
        for (int k = grid.k_margin; k < grid.k_total - grid.k_margin; ++k) {
          // clang-format off
          // cs: sound speed, vv: fluid velocity, ca: Alfvén speed
          Real vv = util::sqrt(
            + qq.vx(i,j,k)*qq.vx(i,j,k)
            + qq.vy(i,j,k)*qq.vy(i,j,k)
            + qq.vz(i,j,k)*qq.vz(i,j,k)
          );
          Real ca = util::sqrt((
            + qq.bx(i,j,k)*qq.bx(i,j,k)
            + qq.by(i,j,k)*qq.by(i,j,k)
            + qq.bz(i,j,k)*qq.bz(i,j,k)
          )/qq.ro(i,j,k)*pii4<Real>);

          Real total_vel = cs(i, j, k) + vv + ca;
          dt = util::min2(dt, cfl_number
            * util::min3<Real>(grid.dx[i], grid.dy[j], grid.dz[k])/total_vel);
          // clang-format on
        }
      }
    }
    Real dt_global;
    MPI_Allreduce(&dt, &dt_global, 1, mpi::data_type<Real>(), MPI_MIN,
                  mpi::comm());
    return dt_global;
  }

  /// @brief Set parameters for divergence B cleaning
  void divb_parameters_set(Real dt) {
    ch_divb = 0.8 * cfl_number * grid.min_dxyz / dt;
    ch_divb_square = ch_divb * ch_divb;
    tau_divb = 2.0 * dt;
  }

  /// @brief Update MHD equations by one time step
  template <typename EOS, typename BoundaryCondition, typename Source>
  void update(Real dt, const EOS &eos, const BoundaryCondition &bc,
              const Source &src, Fields<Real, backend::Host> &qq) {
    divb_parameters_set(dt);
    runge_kutta_4step(dt, eos, bc, src, qq);
    apply_artificial_viscosity(dt, eos, bc, qq);
  }

  // Prohibit copy and move
  Integrator(const Integrator &) = delete;
  Integrator &operator=(const Integrator &) = delete;
  Integrator(Integrator &&) = delete;
  Integrator &operator=(Integrator &&) = delete;
};

}  // namespace mhd
}  // namespace miso
