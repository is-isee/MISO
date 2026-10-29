#pragma once
#include "common.hpp"

/// @brief gx of the stratification at the heights of the cells
inline Array1D<Real, Backend> gravity(const Grid<Real> &grid,
                                      const Stratification<Real> &strat) {
  Array1D<Real> g(grid.k_total);
  for (int k = 0; k < grid.k_total; ++k) {
    g[k] = strat.at(grid.z[k]).gx;
  }
  Array1D<Real, Backend> out(grid.k_total);
  out.copy_from(g);
  return out;
}

/// @brief Damping rate rate ((z - z_edge) / thickness)^2 above z_edge, the
/// thickness below the top (as top_velocity_damping of R2D2plus)
inline Array1D<Real, Backend> top_layer(Config &config, const Grid<Real> &grid) {
  const Real thickness = config["photosphere"]["top_layer"].as<Real>();
  const Real rate = config["photosphere"]["top_layer_rate"].as<Real>();
  const Real z_edge = config["grid"]["z_max"].as<Real>() - thickness;
  Array1D<Real> w(grid.k_total);
  for (int k = 0; k < grid.k_total; ++k) {
    w[k] = grid.z[k] > z_edge
               ? rate * util::pow2((grid.z[k] - z_edge) / thickness)
               : Real(0);
  }
  Array1D<Real, Backend> out(grid.k_total);
  out.copy_from(w);
  return out;
}

/// @brief Gravity, radiative heating and cooling, and the velocity damping.
/// @details It is copied into the kernels, so it holds only views and scalars.
/// The arrays of gx and damping_top are owned by the caller.
struct SourceTerm {
  rt::RadiativeHeatingSource<Real> rad;
  /// @brief Gravitational acceleration [cm/s^2] toward -z, as a function of k
  Array1DView<const Real> gx;
  /// @brief Rate [1/s] of the velocity damping at the start (set by set_time)
  Real damping_init = 0;
  /// @brief Rate [1/s] of the velocity damping near the top, as a function of
  /// k (absorbing layer), and its maximum
  Array1DView<const Real> damping_top;
  Real damping_top_max;
  /// @brief End [s] and initial time scale [s] of the velocity damping
  Real damping_end, damping_tau;

  SourceTerm(Config &config, rt::RadiativeHeatingSource<Real> rad,
             const Array1D<Real, Backend> &gx,
             const Array1D<Real, Backend> &damping_top)
      : rad(rad), gx(gx.const_view()), damping_top(damping_top.const_view()),
        damping_top_max(config["photosphere"]["top_layer_rate"].as<Real>()),
        damping_end(config["photosphere"]["damping_end"].as<Real>()),
        damping_tau(config["photosphere"]["damping_tau"].as<Real>()) {}

  /// @brief Set the damping at the start for the time t [s]. It weakens
  /// linearly and vanishes at damping_end, as [physics.initial_damping] of
  /// R2D2plus.
  void set_time(Real t) {
    damping_init =
        t < damping_end ? (damping_end - t) / damping_end / damping_tau : Real(0);
  }

  /// @brief The damping is explicit: rate * dt <= 0.5
  Real dt_limit() const { return Real(0.5) / (damping_init + damping_top_max); }

  __host__ __device__ Real damping(int k) const {
    return damping_init + damping_top[k];
  }

  __host__ __device__ Real vx(mhd::FieldsView<const Real> qq, int i, int j,
                              int k) const {
    return -damping(k) * qq.ro(i, j, k) * qq.vx(i, j, k);
  }

  __host__ __device__ Real vy(mhd::FieldsView<const Real> qq, int i, int j,
                              int k) const {
    return -damping(k) * qq.ro(i, j, k) * qq.vy(i, j, k);
  }

  __host__ __device__ Real vz(mhd::FieldsView<const Real> qq, int i, int j,
                              int k) const {
    return -(gx[k] + damping(k) * qq.vz(i, j, k)) * qq.ro(i, j, k);
  }

  /// Q_rad and the work of the forces
  __host__ __device__ Real ei(mhd::FieldsView<const Real> qq, int i, int j,
                              int k) const {
    return rad.ei(qq, i, j, k) + vx(qq, i, j, k) * qq.vx(i, j, k) +
           vy(qq, i, j, k) * qq.vy(i, j, k) + vz(qq, i, j, k) * qq.vz(i, j, k);
  }
};
