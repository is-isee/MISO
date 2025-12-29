#pragma once

#include <miso/array3d.hpp>
#include <miso/constants.hpp>
#include <miso/env.hpp>
#include <miso/mhd_artificial_viscosity_cpu.hpp>
#include <miso/mhd_fields.hpp>

namespace miso {
namespace mhd {
namespace impl_host {

/// @brief Calculate 4th order space-centered derivative for qq
template <typename Real>
inline Real space_centered_4th(const Array3D<Real, HostSpace> &qq, Real dxyzi,
                               int i, int j, int k, int is, int js, int ks) {
  // clang-format off
  return (
    -     qq(i + 2*is, j + 2*js, k + 2*ks)
    + 8.0*qq(i +   is, j +   js, k +   ks)
    - 8.0*qq(i -   is, j -   js, k -   ks)
    +     qq(i - 2*is, j - 2*js, k - 2*ks)
  )*inv12<Real>*dxyzi;
  // clang-format on
};

/// @brief Calculate 4th order space-centered derivative for qq1*qq2
template <typename Real>
inline Real space_centered_4th(const Array3D<Real, HostSpace> &qq1,
                               const Array3D<Real, HostSpace> &qq2, Real dxyzi,
                               int i, int j, int k, int is, int js, int ks) {
  // clang-format off
  return (
    -     qq1(i + 2*is, j + 2*js, k + 2*ks)*qq2(i + 2*is, j + 2*js, k + 2*ks)
    + 8.0*qq1(i +   is, j +   js, k +   ks)*qq2(i +   is, j +   js, k +   ks)
    - 8.0*qq1(i -   is, j -   js, k -   ks)*qq2(i -   is, j -   js, k -   ks)
    +     qq1(i - 2*is, j - 2*js, k - 2*ks)*qq2(i - 2*is, j - 2*js, k - 2*ks)
  )*inv12<Real>*dxyzi;
  // clang-format on
};

/// @brief Calculate 4th order space-centered derivative for qq1*qq2*qq3
template <typename Real>
inline Real space_centered_4th(const Array3D<Real, HostSpace> &qq1,
                               const Array3D<Real, HostSpace> &qq2,
                               const Array3D<Real, HostSpace> &qq3, Real dxyzi,
                               int i, int j, int k, int is, int js, int ks) {
  // clang-format off
  return (
    -     qq1(i + 2*is, j + 2*js, k + 2*ks)*qq2(i + 2*is, j + 2*js, k + 2*ks)*qq3(i + 2*is, j + 2*js, k + 2*ks)
    + 8.0*qq1(i +   is, j +   js, k +   ks)*qq2(i +   is, j +   js, k +   ks)*qq3(i +   is, j +   js, k +   ks)
    - 8.0*qq1(i -   is, j -   js, k -   ks)*qq2(i -   is, j -   js, k -   ks)*qq3(i -   is, j -   js, k -   ks)
    +     qq1(i - 2*is, j - 2*js, k - 2*ks)*qq2(i - 2*is, j - 2*js, k - 2*ks)*qq3(i - 2*is, j - 2*js, k - 2*ks)
  )*inv12<Real>*dxyzi;
  // clang-format on
};

/// @brief Dummy source class (without source terms)
/// @details Volumetric heat / force terms are expected.
template <typename Real> struct NoSource {
  /// External force: x-direction
  inline Real vx(const Fields<Real> &, int, int, int) const noexcept {
    return 0.0;
  }

  /// External force: y-direction
  inline Real vy(const Fields<Real> &, int, int, int) const noexcept {
    return 0.0;
  }

  /// External force: z-direction
  inline Real vz(const Fields<Real> &, int, int, int) const noexcept {
    return 0.0;
  }

  /// External heating
  inline Real ei(const Fields<Real> &, int, int, int) const noexcept {
    return 0.0;
  }
};

/// @brief Class for time integration of MHD equations
template <typename Real, typename BoundaryCondition, typename EOS,
          typename Source>
struct Integrator {
  /// @brief Spatial grid
  Grid<Real, HostSpace> &grid;
  /// @brief Equation of states
  EOS eos;
  /// @brief MHD state
  Fields<Real> &qq;
  /// @brief Workspace
  Fields<Real> qq_argm, qq_rslt;

  /// @brief Halo exchanger
  HaloExchanger<Real> halo_exchanger;
  /// @brief Boundary condition for MHD equations
  BoundaryCondition bc;
  /// @brief Body source for MHD equations
  Source source;
  /// @brief Artificial viscosity for MHD equations
  ArtificialViscosity<Real, EOS> artdiff;

  /// @brief gas pressure
  Array3D<Real, HostSpace> pr;
  /// @brief magnetic field strength bx*bx + by*by + bz*bz
  Array3D<Real, HostSpace> bb;
  /// @brief enthalpy + 2*magnetic energy + kinetic energy
  Array3D<Real, HostSpace> ht;
  /// @brief inner product of velocity and magnetic field vx*bx + vy*by + vz*bz
  Array3D<Real, HostSpace> vb;

  /// @brief CFL number
  Real cfl_number;
  /// @brief propagation speed fo divergence B
  Real ch_divb;
  /// @brief square of ch_divb;
  Real ch_divb_square;
  /// @brief damping time scape for divergence B
  Real tau_divb;

  /// @brief Constructor
  Integrator(Config &config, Fields<Real> &qq, Grid<Real, HostSpace> &grid,
             ExecContext &exec_ctx)
      : grid(grid), eos(config), qq(qq), qq_argm(grid), qq_rslt(grid),
        halo_exchanger(grid, exec_ctx), bc(config), artdiff(config, grid, eos),
        pr(grid.i_total, grid.j_total, grid.k_total),
        bb(grid.i_total, grid.j_total, grid.k_total),
        ht(grid.i_total, grid.j_total, grid.k_total),
        vb(grid.i_total, grid.j_total, grid.k_total) {
    cfl_number = config["mhd"]["cfl_number"].template as<Real>();
  }

  /// @brief Update MHD equations using 4th order space-centered scheme
  void update_sc4(Fields<Real> &qq_orgn, Fields<Real> &qq_argm,
                  Fields<Real> &qq_rslt, const Real dt) {
    for (int i = 0; i < grid.i_total; ++i) {
      for (int j = 0; j < grid.j_total; ++j) {
        for (int k = 0; k < grid.k_total; ++k) {
          // clang-format off
          // gas pressure
          pr(i, j, k) = qq_argm.ro(i,j,k)*qq_argm.ei(i,j,k)*(eos.gm - 1.0);
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

    for (int i = grid.i_margin; i < grid.i_total - grid.i_margin; ++i) {
      const Real dxi = grid.dxi[i];
      for (int j = grid.j_margin; j < grid.j_total - grid.j_margin; ++j) {
        const Real dyi = grid.dyi[j];
        for (int k = grid.k_margin; k < grid.k_total - grid.k_margin; ++k) {
          const Real dzi = grid.dzi[k];

          // clang-format off
          // equation of continuity
          qq_rslt.ro(i, j, k) = qq_orgn.ro(i, j, k) + dt * (
          -space_centered_4th(qq_argm.ro, qq_argm.vx, dxi, i, j, k, grid.is, 0, 0)
          -space_centered_4th(qq_argm.ro, qq_argm.vy, dyi, i, j, k, 0, grid.js, 0)
          -space_centered_4th(qq_argm.ro, qq_argm.vz, dzi, i, j, k, 0, 0, grid.ks)
          );

          // x equation of motion
          qq_rslt.vx(i, j, k) = (
              qq_orgn.ro(i, j, k) * qq_orgn.vx(i, j, k) + dt * (
              -space_centered_4th(qq_argm.ro, qq_argm.vx, qq_argm.vx, dxi, i, j, k, grid.is, 0, 0)
              -space_centered_4th(qq_argm.ro, qq_argm.vx, qq_argm.vy, dyi, i, j, k, 0, grid.js, 0)
              -space_centered_4th(qq_argm.ro, qq_argm.vx, qq_argm.vz, dzi, i, j, k, 0, 0, grid.ks)
              -space_centered_4th(pr, dxi, i, j, k, grid.is, 0, 0)
              -space_centered_4th(bb, dxi, i, j, k, grid.is, 0, 0)*pii8<Real>
              +pii4<Real>*(
                  +space_centered_4th(qq_argm.bx, qq_argm.bx, dxi, i, j, k, grid.is, 0, 0)
                  +space_centered_4th(qq_argm.bx, qq_argm.by, dyi, i, j, k, 0, grid.js, 0)
                  +space_centered_4th(qq_argm.bx, qq_argm.bz, dzi, i, j, k, 0, 0, grid.ks)
                  )
              +source.vx(qq_argm, i, j, k)
              )
          )/qq_rslt.ro(i, j, k);

          // y equation of motion
          qq_rslt.vy(i, j, k) = (
              qq_orgn.ro(i, j, k) * qq_orgn.vy(i, j, k) + dt * (
              -space_centered_4th(qq_argm.ro, qq_argm.vy, qq_argm.vx, dxi, i, j, k, grid.is, 0, 0)
              -space_centered_4th(qq_argm.ro, qq_argm.vy, qq_argm.vy, dyi, i, j, k, 0, grid.js, 0)
              -space_centered_4th(qq_argm.ro, qq_argm.vy, qq_argm.vz, dzi, i, j, k, 0, 0, grid.ks)
              -space_centered_4th(pr, dyi, i, j, k, 0, grid.js, 0)
              -space_centered_4th(bb, dyi, i, j, k, 0, grid.js, 0)*pii8<Real>
              +pii4<Real>*(
                  +space_centered_4th(qq_argm.by, qq_argm.bx, dxi, i, j, k, grid.is, 0, 0)
                  +space_centered_4th(qq_argm.by, qq_argm.by, dyi, i, j, k, 0, grid.js, 0)
                  +space_centered_4th(qq_argm.by, qq_argm.bz, dzi, i, j, k, 0, 0, grid.ks)
                  )
              +source.vy(qq_argm, i, j, k)
              )
          )/qq_rslt.ro(i, j, k);

          // z equation of motion
          qq_rslt.vz(i, j, k) = (
              qq_orgn.ro(i, j, k) * qq_orgn.vz(i, j, k) + dt * (
              -space_centered_4th(qq_argm.ro, qq_argm.vz, qq_argm.vx, dxi, i, j, k, grid.is, 0, 0)
              -space_centered_4th(qq_argm.ro, qq_argm.vz, qq_argm.vy, dyi, i, j, k, 0, grid.js, 0)
              -space_centered_4th(qq_argm.ro, qq_argm.vz, qq_argm.vz, dzi, i, j, k, 0, 0, grid.ks)
              -space_centered_4th(pr, dzi, i, j, k, 0, 0, grid.ks)
              -space_centered_4th(bb, dzi, i, j, k, 0, 0, grid.ks)*pii8<Real>
              +pii4<Real>*(
                  +space_centered_4th(qq_argm.bz, qq_argm.bx, dxi, i, j, k, grid.is, 0, 0)
                  +space_centered_4th(qq_argm.bz, qq_argm.by, dyi, i, j, k, 0, grid.js, 0)
                  +space_centered_4th(qq_argm.bz, qq_argm.bz, dzi, i, j, k, 0, 0, grid.ks)
                  )
              +source.vz(qq_argm, i, j, k)
              )
          )/qq_rslt.ro(i, j, k);

          // x magnetic induction
          qq_rslt.bx(i, j, k) = qq_orgn.bx(i, j, k) + dt * (
              -space_centered_4th(qq_argm.vy, qq_argm.bx, dyi, i, j, k, 0, grid.js, 0)
              -space_centered_4th(qq_argm.vz, qq_argm.bx, dzi, i, j, k, 0, 0, grid.ks)
              +space_centered_4th(qq_argm.vx, qq_argm.by, dyi, i, j, k, 0, grid.js, 0)
              +space_centered_4th(qq_argm.vx, qq_argm.bz, dzi, i, j, k, 0, 0, grid.ks)
              -space_centered_4th(qq_argm.ph, dxi, i, j, k, grid.is, 0, 0)
          );

          // y magnetic induction
          qq_rslt.by(i, j, k) = qq_orgn.by(i, j, k) + dt * (
              -space_centered_4th(qq_argm.vx, qq_argm.by, dxi, i, j, k, grid.is, 0, 0)
              -space_centered_4th(qq_argm.vz, qq_argm.by, dzi, i, j, k, 0, 0, grid.ks)
              +space_centered_4th(qq_argm.vy, qq_argm.bx, dxi, i, j, k, grid.is, 0, 0)
              +space_centered_4th(qq_argm.vy, qq_argm.bz, dzi, i, j, k, 0, 0, grid.ks)
              -space_centered_4th(qq_argm.ph, dyi, i, j, k, 0, grid.js, 0)
          );

          // z magnetic induction
          qq_rslt.bz(i, j, k) = qq_orgn.bz(i, j, k) + dt * (
              -space_centered_4th(qq_argm.vx, qq_argm.bz, dxi, i, j, k, grid.is, 0, 0)
              -space_centered_4th(qq_argm.vy, qq_argm.bz, dyi, i, j, k, 0, grid.js, 0)
              +space_centered_4th(qq_argm.vz, qq_argm.bx, dxi, i, j, k, grid.is, 0, 0)
              +space_centered_4th(qq_argm.vz, qq_argm.by, dyi, i, j, k, 0, grid.js, 0)
              -space_centered_4th(qq_argm.ph, dzi, i, j, k, 0, 0, grid.ks)
          );

          // div B factor
          qq_rslt.ph(i, j, k) = (
              qq_orgn.ph(i, j, k) + dt * ch_divb_square*(
              -space_centered_4th(qq_argm.bx, dxi, i, j, k, grid.is, 0, 0)
              -space_centered_4th(qq_argm.by, dyi, i, j, k, 0, grid.js, 0)
              -space_centered_4th(qq_argm.bz, dzi, i, j, k, 0, 0, grid.ks)
              )
          )*std::exp(-dt/tau_divb);

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
              -space_centered_4th(ht, qq_argm.vx, dxi, i, j, k, grid.is, 0, 0)
              -space_centered_4th(ht, qq_argm.vy, dyi, i, j, k, 0, grid.js, 0)
              -space_centered_4th(ht, qq_argm.vz, dzi, i, j, k, 0, 0, grid.ks)
              + pii4<Real>* (
                  +space_centered_4th(vb, qq_argm.bx, dxi, i, j, k, grid.is, 0, 0)
                  +space_centered_4th(vb, qq_argm.by, dyi, i, j, k, 0, grid.js, 0)
                  +space_centered_4th(vb, qq_argm.bz, dzi, i, j, k, 0, 0, grid.ks)
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
              + source.ei(qq_argm, i, j, k)
          )/qq_rslt.ro(i, j, k);
          // clang-format on
        }
      }
    }
  }

  /// @brief Runge-Kutta 4th order time integration step
  void runge_kutta_4step(const Real dt) {
    update_sc4(qq, qq, qq_rslt, dt / 4.0);
    qq_argm.copy_from(qq_rslt);
    bc.apply(qq_argm.view());
    halo_exchanger.apply(qq_argm);

    update_sc4(qq, qq_argm, qq_rslt, dt / 3.0);
    qq_argm.copy_from(qq_rslt);
    bc.apply(qq_argm.view());
    halo_exchanger.apply(qq_argm);

    update_sc4(qq, qq_argm, qq_rslt, dt / 2.0);
    qq_argm.copy_from(qq_rslt);
    bc.apply(qq_argm.view());
    halo_exchanger.apply(qq_argm);

    update_sc4(qq, qq_argm, qq_rslt, dt);
    qq.copy_from(qq_rslt);
    bc.apply(qq.view());
    halo_exchanger.apply(qq);
  }

  /// @brief Apply artificial viscosity
  void apply_artificial_viscosity(const Real dt) {
    artdiff.characteristic_velocity_eval(qq);

    // x direction
    artdiff.update(qq, qq_rslt, Direction::X, dt);
    qq.copy_from(qq_rslt);
    bc.apply(qq.view());
    halo_exchanger.apply(qq);

    // y direction
    artdiff.update(qq, qq_rslt, Direction::Y, dt);
    qq.copy_from(qq_rslt);
    bc.apply(qq.view());
    halo_exchanger.apply(qq);

    // z direction
    artdiff.update(qq, qq_rslt, Direction::Z, dt);
    qq.copy_from(qq_rslt);
    bc.apply(qq.view());
    halo_exchanger.apply(qq);
  }

  /// @brief Calculate time spacing based on CFL condition
  Real cfl() const {
    Real dt = 1.e10;
    for (int i = grid.i_margin; i < grid.i_total - grid.i_margin; ++i) {
      for (int j = grid.j_margin; j < grid.j_total - grid.j_margin; ++j) {
        for (int k = grid.k_margin; k < grid.k_total - grid.k_margin; ++k) {
          // clang-format off
          // cs: sound speed, vv: fluid velocity, ca: Alfvén speed
          Real cs = std::sqrt(eos.gm*(eos.gm-1.0)*qq.ei(i,j,k));
          Real vv = std::sqrt(
            + qq.vx(i,j,k)*qq.vx(i,j,k)
            + qq.vy(i,j,k)*qq.vy(i,j,k)
            + qq.vz(i,j,k)*qq.vz(i,j,k)
          );
          Real ca = std::sqrt( (
            + qq.bx(i,j,k)*qq.bx(i,j,k)
            + qq.by(i,j,k)*qq.by(i,j,k)
            + qq.bz(i,j,k)*qq.bz(i,j,k)
          )/qq.ro(i,j,k)*pii4<Real>);

          Real total_vel = cs + vv + ca;
          dt = std::min(dt,
            cfl_number*std::min<Real>({grid.dx[i], grid.dy[j], grid.dz[k]})/total_vel);
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
  void divb_parameters_set(const Real dt) {
    ch_divb = 0.8 * cfl_number * grid.min_dxyz / dt;
    ch_divb_square = ch_divb * ch_divb;
    tau_divb = 2.0 * dt;
  }

  /// @brief Update MHD equations by one time step
  void update(const Real dt) {
    divb_parameters_set(dt);
    runge_kutta_4step(dt);
    apply_artificial_viscosity(dt);
  }

  // Prohibit copy and move
  Integrator(const Integrator &) = delete;
  Integrator &operator=(const Integrator &) = delete;
  Integrator(Integrator &&) = delete;
  Integrator &operator=(Integrator &&) = delete;
};

}  // namespace impl_host
}  // namespace mhd
}  // namespace miso
