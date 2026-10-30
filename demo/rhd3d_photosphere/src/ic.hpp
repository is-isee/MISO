#pragma once
#include <random>

#include "common.hpp"

/// @brief The 1D stratification and a small random vertical velocity
struct InitialCondition {
  const Stratification<Real> &strat;
  /// @brief Amplitude of the random vertical velocity [cm/s]
  Real v_amp;

  InitialCondition(Config &config, const Stratification<Real> &strat)
      : strat(strat), v_amp(config["photosphere"]["v_amp"].as<Real>()) {}

  // The signature must not be changed as it is called inside miso::mhd::MHD.
  void apply(mhd::FieldsView<Real> qq, GridView<const Real> grid) const {
    std::mt19937 engine(mpi::rank());
    std::uniform_real_distribution<Real> dist(-1.0, 1.0);
    for (int k = 0; k < grid.k_total; ++k) {
      const auto s = strat.at(grid.z[k]);
      for (int i = 0; i < grid.i_total; ++i) {
        for (int j = 0; j < grid.j_total; ++j) {
          qq.ro(i, j, k) = s.ro;
          qq.ei(i, j, k) = s.ei;
          qq.vx(i, j, k) = 0.0;
          qq.vy(i, j, k) = 0.0;
          qq.vz(i, j, k) = v_amp * dist(engine);
          qq.bx(i, j, k) = 0.0;
          qq.by(i, j, k) = 0.0;
          qq.bz(i, j, k) = 0.0;
          qq.ph(i, j, k) = 0.0;
        }
      }
    }
  }
};
