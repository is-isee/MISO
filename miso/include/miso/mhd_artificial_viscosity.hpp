#pragma once

#include <stdexcept>

#include "array3d.hpp"
#include "config.hpp"
#include "constants.hpp"
#include "cuda_compat.hpp"
#include "execution.hpp"
#include "grid.hpp"
#include "limiter.hpp"
#include "mhd_fields.hpp"
#include "types.hpp"

namespace miso {
namespace mhd {

/// @brief Artificial viscosity flux in one direction at a cell.
/// @details Written once for all backends: every member function is
/// __host__ __device__ and called from for_each. The struct holds views only,
/// so it is cheap to copy into a CUDA kernel. Each function writes one
/// variable of qq_rslt at (i, j, k); vx, vy, vz and ei read qq_rslt.ro at the
/// same cell, and ei also reads qq_rslt.vx, ..., bz, so call ro first and ei
/// last.
template <typename Real> struct AvKernel {
  FieldsView<const Real> qq;
  FieldsView<Real> qq_rslt;
  Array3DView<const Real> cc;
  /// @brief Inverse grid spacing in the direction (dxi, dyi or dzi)
  const Real *dxyzi;
  /// @brief Unit offset in the direction (one of them is 1, others are 0)
  int is, js, ks;
  Real ep, fh, dt;

  /// @brief Difference of the fluxes at the upwind and downwind faces of a
  /// conserved variable q(i, j, k), multiplied by dxyzi * dt.
  template <typename Q>
  __host__ __device__ Real flux_difference(const Q &q, int i, int j,
                                           int k) const {
    using miso::limiter::dqq_eval;
    using miso::limiter::flux_core;
    // characteristic velocity
    // clang-format off
    const Real ccl = cc(i - is, j - js, k - ks);
    const Real ccc = cc(i     , j     , k     );
    const Real ccr = cc(i + is, j + js, k + ks);

    const Real qql2 = q(i - 2 * is, j - 2 * js, k - 2 * ks);
    const Real qql1 = q(i -     is, j -     js, k -     ks);
    const Real qqc  = q(i         , j         , k         );
    const Real qqr1 = q(i +     is, j +     js, k +     ks);
    const Real qqr2 = q(i + 2 * is, j + 2 * js, k + 2 * ks);
    // clang-format on
    // dqq at i-is, j-js, k-ks
    const Real dqq_dw = dqq_eval(qql2, qql1, qqc, ep);
    // dqq at i, j, k
    const Real dqq_cn = dqq_eval(qql1, qqc, qqr1, ep);
    // dqq at i+is, j+js, k+ks
    const Real dqq_up = dqq_eval(qqc, qqr1, qqr2, ep);
    const Real f_dw =
        flux_core(qql1, qqc, dqq_dw, dqq_cn, Real(0.5) * (ccl + ccc), fh);
    const Real f_up =
        flux_core(qqc, qqr1, dqq_cn, dqq_up, Real(0.5) * (ccc + ccr), fh);
    return (f_up - f_dw) * dxyzi[i * is + j * js + k * ks] * dt;
  }

  /// @brief Density
  __host__ __device__ void ro(int i, int j, int k) const {
    const auto q = [&](int i, int j, int k) { return qq.ro(i, j, k); };
    qq_rslt.ro(i, j, k) = qq.ro(i, j, k) - flux_difference(q, i, j, k);
  }

  /// @brief x momentum
  __host__ __device__ void vx(int i, int j, int k) const {
    const auto q = [&](int i, int j, int k) {
      return qq.ro(i, j, k) * qq.vx(i, j, k);
    };
    qq_rslt.vx(i, j, k) =
        (qq.ro(i, j, k) * qq.vx(i, j, k) - flux_difference(q, i, j, k)) /
        qq_rslt.ro(i, j, k);
  }

  /// @brief y momentum
  __host__ __device__ void vy(int i, int j, int k) const {
    const auto q = [&](int i, int j, int k) {
      return qq.ro(i, j, k) * qq.vy(i, j, k);
    };
    qq_rslt.vy(i, j, k) =
        (qq.ro(i, j, k) * qq.vy(i, j, k) - flux_difference(q, i, j, k)) /
        qq_rslt.ro(i, j, k);
  }

  /// @brief z momentum
  __host__ __device__ void vz(int i, int j, int k) const {
    const auto q = [&](int i, int j, int k) {
      return qq.ro(i, j, k) * qq.vz(i, j, k);
    };
    qq_rslt.vz(i, j, k) =
        (qq.ro(i, j, k) * qq.vz(i, j, k) - flux_difference(q, i, j, k)) /
        qq_rslt.ro(i, j, k);
  }

  /// @brief x magnetic field
  __host__ __device__ void bx(int i, int j, int k) const {
    const auto q = [&](int i, int j, int k) { return qq.bx(i, j, k); };
    qq_rslt.bx(i, j, k) = qq.bx(i, j, k) - flux_difference(q, i, j, k);
  }

  /// @brief y magnetic field
  __host__ __device__ void by(int i, int j, int k) const {
    const auto q = [&](int i, int j, int k) { return qq.by(i, j, k); };
    qq_rslt.by(i, j, k) = qq.by(i, j, k) - flux_difference(q, i, j, k);
  }

  /// @brief z magnetic field
  __host__ __device__ void bz(int i, int j, int k) const {
    const auto q = [&](int i, int j, int k) { return qq.bz(i, j, k); };
    qq_rslt.bz(i, j, k) = qq.bz(i, j, k) - flux_difference(q, i, j, k);
  }

  /// @brief div B cleaning potential
  __host__ __device__ void ph(int i, int j, int k) const {
    const auto q = [&](int i, int j, int k) { return qq.ph(i, j, k); };
    qq_rslt.ph(i, j, k) = qq.ph(i, j, k) - flux_difference(q, i, j, k);
  }

  /// @brief Total energy
  /// @details Et: total energy per unit volume, ei: internal energy per unit
  /// mass
  __host__ __device__ void ei(int i, int j, int k) const {
    const auto q = [&](int i, int j, int k) {
      return qq.ro(i, j, k) * qq.ei(i, j, k);
    };
    const Real Et =
        qq.ro(i, j, k) * qq.ei(i, j, k) +
        Real(0.5) * qq.ro(i, j, k) *
            (qq.vx(i, j, k) * qq.vx(i, j, k) + qq.vy(i, j, k) * qq.vy(i, j, k) +
             qq.vz(i, j, k) * qq.vz(i, j, k)) +
        pii8<Real> *
            (qq.bx(i, j, k) * qq.bx(i, j, k) + qq.by(i, j, k) * qq.by(i, j, k) +
             qq.bz(i, j, k) * qq.bz(i, j, k));

    qq_rslt.ei(i, j, k) =
        (Et - flux_difference(q, i, j, k) -
         Real(0.5) * qq_rslt.ro(i, j, k) *
             (qq_rslt.vx(i, j, k) * qq_rslt.vx(i, j, k) +
              qq_rslt.vy(i, j, k) * qq_rslt.vy(i, j, k) +
              qq_rslt.vz(i, j, k) * qq_rslt.vz(i, j, k)) -
         pii8<Real> * (qq_rslt.bx(i, j, k) * qq_rslt.bx(i, j, k) +
                       qq_rslt.by(i, j, k) * qq_rslt.by(i, j, k) +
                       qq_rslt.bz(i, j, k) * qq_rslt.bz(i, j, k))) /
        qq_rslt.ro(i, j, k);
  }
};

/// @brief Artificial viscosity for MHD equations
/// @details One implementation for all backends (see Integrator).
template <typename Real, typename Backend> struct ArtificialViscosity {
  Grid<Real, Backend> &grid;

  /// @brief Characteristic velocity cs_fac*cs + ca_fac*ca + vv_fac*vv
  Array3D<Real, Backend> cc;
  /// @brief Speed of sound
  Array3D<Real, Backend> cs;
  /// @brief Parameters for generalized minmod limiter
  Real ep;
  /// @brief Parameters for amplitude of artificial viscosity flux
  Real fh;
  /// @brief Characteristic velocity factor for sound speed
  Real cs_fac;
  /// @brief Characteristic velocity factor for Alfvén speed
  Real ca_fac;
  /// @brief Characteristic velocity factor for fluid velocity
  Real vv_fac;

  /// @brief Constructor for ArtificialViscosity
  ArtificialViscosity(Config &config, Grid<Real, Backend> &grid)
      : grid(grid), cc(grid.i_total, grid.j_total, grid.k_total),
        cs(grid.i_total, grid.j_total, grid.k_total) {
    ep = config["mhd"]["artificial_viscosity"]["ep"].as<Real>();
    fh = config["mhd"]["artificial_viscosity"]["fh"].as<Real>();
    cs_fac = config["mhd"]["artificial_viscosity"]["cs_fac"].as<Real>();
    ca_fac = config["mhd"]["artificial_viscosity"]["ca_fac"].as<Real>();
    vv_fac = config["mhd"]["artificial_viscosity"]["vv_fac"].as<Real>();
    if (!(ep >= 0)) {
      throw std::runtime_error(
          "mhd.artificial_viscosity.ep must be non-negative.");
    }
    if (!(fh >= 0)) {
      throw std::runtime_error(
          "mhd.artificial_viscosity.fh must be non-negative.");
    }
  }

  /// @brief Evaluate the characteristic velocity
  template <typename EOS>
  void characteristic_velocity_eval(const Fields<Real, Backend> &qq,
                                    const EOS &eos) {
    const auto qq_v = qq.const_view();
    const auto cs_v = cs.const_view();
    const auto cc_v = cc.view();
    // device lambdas must not capture `this`
    const Real cs_fac_ = cs_fac, ca_fac_ = ca_fac, vv_fac_ = vv_fac;

    eos.sound_speed(Backend{}, qq_v, cs.view());
    const Range3D range{{0, grid.i_total}, {0, grid.j_total}, {0, grid.k_total}};
    for_each(
        Backend{}, range, MISO_LAMBDA(int i, int j, int k) {
          // cs: sound speed, vv: fluid velocity, ca: Alfvén speed
          Real vv = util::sqrt(qq_v.vx(i, j, k) * qq_v.vx(i, j, k) +
                               qq_v.vy(i, j, k) * qq_v.vy(i, j, k) +
                               qq_v.vz(i, j, k) * qq_v.vz(i, j, k));
          Real ca = util::sqrt((qq_v.bx(i, j, k) * qq_v.bx(i, j, k) +
                                qq_v.by(i, j, k) * qq_v.by(i, j, k) +
                                qq_v.bz(i, j, k) * qq_v.bz(i, j, k)) /
                               qq_v.ro(i, j, k) * pii4<Real>);
          cc_v(i, j, k) = cs_v(i, j, k) * cs_fac_ + vv * vv_fac_ + ca * ca_fac_;
        });
  }

  /// @brief Update qq_rslt by the artificial viscosity flux in one direction
  void update(const Fields<Real, Backend> &qq, Fields<Real, Backend> &qq_rslt,
              Direction direction, const Real dt) {
    const auto grid_v = grid.const_view();
    Range3D range{{0, grid.i_total}, {0, grid.j_total}, {0, grid.k_total}};
    int is = 0, js = 0, ks = 0;
    const Real *dxyzi = nullptr;
    if (direction == Direction::X) {
      range.range0 = {2 * grid.is, grid.i_total - 2 * grid.is};
      is = grid.is;
      dxyzi = grid_v.dxi;
    } else if (direction == Direction::Y) {
      range.range1 = {2 * grid.js, grid.j_total - 2 * grid.js};
      js = grid.js;
      dxyzi = grid_v.dyi;
    } else if (direction == Direction::Z) {
      range.range2 = {2 * grid.ks, grid.k_total - 2 * grid.ks};
      ks = grid.ks;
      dxyzi = grid_v.dzi;
    }

    const AvKernel<Real> kern{qq.const_view(),
                              qq_rslt.view(),
                              cc.const_view(),
                              dxyzi,
                              is,
                              js,
                              ks,
                              ep,
                              fh,
                              dt};
    // All variables in one loop (faster on RTX 3090 than one loop per
    // variable, #197).
    for_each(
        Backend{}, range, MISO_LAMBDA(int i, int j, int k) {
          kern.ro(i, j, k);
          kern.vx(i, j, k);
          kern.vy(i, j, k);
          kern.vz(i, j, k);
          kern.bx(i, j, k);
          kern.by(i, j, k);
          kern.bz(i, j, k);
          kern.ph(i, j, k);
          kern.ei(i, j, k);
        });
  }
};

}  // namespace mhd
}  // namespace miso
