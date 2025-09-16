#pragma once

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <initializer_list>
#include <mpi.h>

#include "artificial_viscosity_cpu.hpp"
#include "constants.hpp"
#include "custom_boundary_condition.hpp"
#include "model.hpp"
#include "mpi_types.hpp"
#include "standard_boundary_condition.hpp"

/// @brief Calculate 4th order space-centered derivative for qq
/// @relatedalso TimeIntegrator
/// @tparam Real
/// @param qq variables
/// @param dxyzi grid spacing in x, y, z direction (grid.dx, grid.dy, grid.dz)
/// @param i i index
/// @param j j index
/// @param k k index
/// @param is i skip
/// @param js j skip
/// @param ks k skip
/// @return Array3D<Real> derivative value
template <typename Real>
inline Real space_centered_4th(const Array3D<Real> &qq, Real dxyzi, int i, int j,
                               int k, int is, int js, int ks) {
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
/// @relatedalso TimeIntegrator
/// @tparam Real
/// @param qq1 variables
/// @param qq2 variables
/// @param dxyzi grid spacing in x, y, z direction (grid.dx, grid.dy, grid.dz)
/// @param i i index
/// @param j j index
/// @param k k index
/// @param is i skip
/// @param js j skip
/// @param ks k skip
/// @return Array3D<Real> derivative value
template <typename Real>
inline Real space_centered_4th(const Array3D<Real> &qq1, const Array3D<Real> &qq2,
                               Real dxyzi, int i, int j, int k, int is, int js,
                               int ks) {
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
/// @relatedalso TimeIntegrator
/// @tparam Real
/// @param qq1 variables
/// @param qq2 variables
/// @param qq3 variables
/// @param dxyzi grid spacing in x, y, z direction (grid.dx, grid.dy, grid.dz)
/// @param i i index
/// @param j j index
/// @param k k index
/// @param is i skip
/// @param js j skip
/// @param ks k skip
/// @return Array3D<Real> derivative value
template <typename Real>
inline Real space_centered_4th(const Array3D<Real> &qq1, const Array3D<Real> &qq2,
                               const Array3D<Real> &qq3, Real dxyzi, int i, int j,
                               int k, int is, int js, int ks) {
  // clang-format off
  return (
    -     qq1(i + 2*is, j + 2*js, k + 2*ks)*qq2(i + 2*is, j + 2*js, k + 2*ks)*qq3(i + 2*is, j + 2*js, k + 2*ks)
    + 8.0*qq1(i +   is, j +   js, k +   ks)*qq2(i +   is, j +   js, k +   ks)*qq3(i +   is, j +   js, k +   ks)
    - 8.0*qq1(i -   is, j -   js, k -   ks)*qq2(i -   is, j -   js, k -   ks)*qq3(i -   is, j -   js, k -   ks)
    +     qq1(i - 2*is, j - 2*js, k - 2*ks)*qq2(i - 2*is, j - 2*js, k - 2*ks)*qq3(i - 2*is, j - 2*js, k - 2*ks)
  )*inv12<Real>*dxyzi;
  // clang-format on
};

/// @brief Class for time integration of MHD equations
/// @tparam Real
template <typename Real> struct TimeIntegrator {
  /// @brief Disallow copying and assignment
  TimeIntegrator(const TimeIntegrator &) = delete;
  /// @brief Disallow copying and assignment
  TimeIntegrator &operator=(const TimeIntegrator &) = delete;
  /// @brief Model class object
  Model<Real> &model;
  /// @brief Config class object
  Config &config;
  /// @brief Time class object
  Time<Real> &time;
  /// @brief Grid class object
  Grid<Real> &grid;
  /// @brief EOS class object
  EOS<Real> &eos;
  /// @brief MHD class object
  MHD<Real> &mhd;
  /// @brief MPIManager class object
  MPIManager<Real> &mpi;

  /// @brief Boundary condition for MHD equations
  std::unique_ptr<BoundaryConditionBase<Real, MHDCore<Real>, Grid<Real>>> bc;
  /// @brief Artificial viscosity for MHD equations
  ArtificialViscosity<Real> artdiff;

  /// @brief gas pressure
  Array3D<Real> pr;
  /// @brief magnetic field strength bx*bx + by*by + bz*bz
  Array3D<Real> bb;
  /// @brief enthalpy + 2*magnetic energy + kinetic energy
  Array3D<Real> ht;
  /// @brief inner product of velocity and magnetic field vx*bx + vy*by + vz*bz
  Array3D<Real> vb;
  /// @brief CFL number
  Real cfl_number;
  /// @brief propagation speed fo divergence B
  Real ch_divb;
  /// @brief square of ch_divb;
  Real ch_divb_square;
  /// @brief damping time scape for divergence B
  Real tau_divb;

  /// @brief Constructor for TimeIntegrator
  /// @param model_
  TimeIntegrator(Model<Real> &model_)
      : model(model_), config(model_.config), time(model_.time),
        grid(model_.grid_local), eos(model_.eos), mhd(model_.mhd),
        artdiff(model_), mpi(model_.mpi),
        pr(grid.i_total, grid.j_total, grid.k_total),
        bb(grid.i_total, grid.j_total, grid.k_total),
        ht(grid.i_total, grid.j_total, grid.k_total),
        vb(grid.i_total, grid.j_total, grid.k_total) {

    if (config.yaml_obj["boundary_condition"]["boundary_type"]
            .template as<std::string>() == "standard") {
      bc = std::make_unique<
          StandardBoundaryCondition<Real, MHDCore<Real>, Grid<Real>>>(model);
    } else if (config.yaml_obj["boundary_condition"]["boundary_type"]
                   .template as<std::string>() == "custom") {
      bc = create_custom_boundary_condition<Real>(model);
    }
    cfl_number =
        config.yaml_obj["time_integrator"]["cfl_number"].template as<Real>();
  }

  /// @brief  Update MHD equations using 4th order space-centered scheme
  /// @param qq_orgn original MHD core variables (n = n)
  /// @param qq_argm argument MHD core variables (n = n + 1/2, n + 1/3, etc.)
  /// @param qq_rslt resulting MHD core variables (n = n + 1)
  /// @param dt time spacing (note that this is not the same as actual time spacing)
  void update_sc4(MHDCore<Real> &qq_orgn, MHDCore<Real> &qq_argm,
                  MHDCore<Real> &qq_rslt, Real dt) {
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
              -0.5*qq_rslt.ro(i,j,k)*(
                  + qq_rslt.vx(i,j,k)*qq_rslt.vx(i,j,k)
                  + qq_rslt.vy(i,j,k)*qq_rslt.vy(i,j,k)
                  + qq_rslt.vz(i,j,k)*qq_rslt.vz(i,j,k) )
              - pii8<Real>*(
                  + qq_rslt.bx(i,j,k)*qq_rslt.bx(i,j,k)
                  + qq_rslt.by(i,j,k)*qq_rslt.by(i,j,k)
                  + qq_rslt.bz(i,j,k)*qq_rslt.bz(i,j,k) )
          )/qq_rslt.ro(i, j, k);
          // clang-format on
        }
      }
    }
  }

  /// @brief Runge-Kutta 4th order time integration step
  void runge_kutta_4step() {
    MHDCore<Real> &qq = mhd.qq;
    MHDCore<Real> &qq_argm = mhd.qq_argm;
    MHDCore<Real> &qq_rslt = mhd.qq_rslt;

    update_sc4(qq, qq, qq_rslt, time.dt / 4.0);
    qq_argm.copy_from(qq_rslt);
    bc->apply(qq_argm);
    mhd.mpi_exchange_halo(qq_argm, grid, mpi);

    update_sc4(qq, qq_argm, qq_rslt, time.dt / 3.0);
    qq_argm.copy_from(qq_rslt);
    bc->apply(qq_argm);
    mhd.mpi_exchange_halo(qq_argm, grid, mpi);

    update_sc4(qq, qq_argm, qq_rslt, time.dt / 2.0);
    qq_argm.copy_from(qq_rslt);
    bc->apply(qq_argm);
    mhd.mpi_exchange_halo(qq_argm, grid, mpi);

    update_sc4(qq, qq_argm, qq_rslt, time.dt);
    qq.copy_from(qq_rslt);
    bc->apply(qq);
    mhd.mpi_exchange_halo(qq, grid, mpi);
  }

  /// @brief Apply artificial viscosity
  void apply_artificial_viscosity() {
    artdiff.characteristic_velocity_eval();

    // x direction
    artdiff.update(mhd.qq, mhd.qq_rslt, artdiff.cc, grid.dxi, "x");
    mhd.qq.copy_from(mhd.qq_rslt);
    bc->apply(mhd.qq);
    mhd.mpi_exchange_halo(mhd.qq, grid, mpi);

    // y direction
    artdiff.update(mhd.qq, mhd.qq_rslt, artdiff.cc, grid.dyi, "y");
    mhd.qq.copy_from(mhd.qq_rslt);
    bc->apply(mhd.qq);
    mhd.mpi_exchange_halo(mhd.qq, grid, mpi);

    // z direction
    artdiff.update(mhd.qq, mhd.qq_rslt, artdiff.cc, grid.dzi, "z");
    mhd.qq.copy_from(mhd.qq_rslt);
    bc->apply(mhd.qq);
    mhd.mpi_exchange_halo(mhd.qq, grid, mpi);
  }

  /// @brief Calculate time spacing based on CFL condition
  void cfl_condition() {

    this->time.dt = 1.e10;

    for (int i = grid.i_margin; i < grid.i_total - grid.i_margin; ++i) {
      for (int j = grid.j_margin; j < grid.j_total - grid.j_margin; ++j) {
        for (int k = grid.k_margin; k < grid.k_total - grid.k_margin; ++k) {
          // clang-format off
          // cs: sound speed, vv: fluid velocity, ca: Alfvén speed
          Real cs = std::sqrt(this->eos.gm*(this->eos.gm-1.0)*this->mhd.qq.ei(i,j,k));
          Real vv = std::sqrt(
            + this->mhd.qq.vx(i,j,k)*this->mhd.qq.vx(i,j,k)
            + this->mhd.qq.vy(i,j,k)*this->mhd.qq.vy(i,j,k)
            + this->mhd.qq.vz(i,j,k)*this->mhd.qq.vz(i,j,k)
          );
          Real ca = std::sqrt( (
            + this->mhd.qq.bx(i,j,k)*this->mhd.qq.bx(i,j,k)
            + this->mhd.qq.by(i,j,k)*this->mhd.qq.by(i,j,k)
            + this->mhd.qq.bz(i,j,k)*this->mhd.qq.bz(i,j,k)
          )/this->mhd.qq.ro(i,j,k)*pii4<Real>);
          this->time.dt = std::min(this->time.dt,
            this->cfl_number*std::min<Real>({this->grid.dx[i], this->grid.dy[j], this->grid.dz[k]})/(cs + vv + ca));
          // clang-format on
        }
      }
    }
    Real dt_global;
    MPI_Allreduce(&this->time.dt, &dt_global, 1, mpi_type<Real>(), MPI_MIN,
                  mpi.cart_comm);
    this->time.dt = dt_global;
  }

  /// @brief Set parameters for divergence B cleaning
  void divb_parameters_set() {
    this->ch_divb = 0.8 * this->cfl_number * this->grid.min_dxyz / this->time.dt;
    this->ch_divb_square = this->ch_divb * this->ch_divb;
    this->tau_divb = 2.0 * this->time.dt;
  }

  /// @brief Main time integration loop
  void run() {
    // Time integration loop
    this->time.dt = 0.1;

    if (this->config.yaml_obj["base"]["continue"].template as<bool>() &&
        fs::exists(this->config.time_save_dir + "n_output.txt")) {
      model.load_state();
    }

    model.save_if_needed();
    while (this->time.time < this->time.tend) {

      // basic MHD time integration
      cfl_condition();
      divb_parameters_set();
      runge_kutta_4step();
      apply_artificial_viscosity();

      // Time is update after all procedures
      this->time.update();
      model.save_if_needed();
    }
  }
};
