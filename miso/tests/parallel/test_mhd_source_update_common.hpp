#pragma once

#include <string>

#include <doctest/doctest.h>

#include <miso/core.hpp>
#include <miso/mhd.hpp>

// Integration test of the source terms through the MHD update.
//
// The initial state is uniform (v = 0, B = 0) with periodic boundaries, so
// that all flux terms and the artificial viscosity vanish. With a constant
// source S, one Runge-Kutta step then gives exactly q = q0 + dt * S, which
// checks that the source terms are scaled by the time step.

namespace test_source_update {

using Real = double;
using namespace miso;

constexpr Real ro0 = 1.0;
constexpr Real pr0 = 1.0;
constexpr Real heating = 2.5;

struct UniformState {
  eos::IdealEOS<Real> &eos;
  void apply(mhd::FieldsView<Real> qq, GridView<const Real>) const {
    for (int n = 0; n < qq.size(); ++n) {
      qq.ro[n] = ro0;
      qq.vx[n] = qq.vy[n] = qq.vz[n] = 0.0;
      qq.bx[n] = qq.by[n] = qq.bz[n] = 0.0;
      qq.ei[n] = pr0 / (eos.gm - 1.0) / ro0;
      qq.ph[n] = 0.0;
    }
  }
};

/// @brief Constant heating (energy per unit volume per unit time)
struct HeatingSource {
  __host__ __device__ Real vx(mhd::FieldsView<const Real>, int, int, int) const {
    return 0.0;
  }
  __host__ __device__ Real vy(mhd::FieldsView<const Real>, int, int, int) const {
    return 0.0;
  }
  __host__ __device__ Real vz(mhd::FieldsView<const Real>, int, int, int) const {
    return 0.0;
  }
  __host__ __device__ Real ei(mhd::FieldsView<const Real>, int, int, int) const {
    return heating;
  }
};

template <typename Backend>
struct Model : public mhd::ModelBase<Model<Backend>, Real, Backend> {
  eos::IdealEOS<Real> eos;
  UniformState ic;
  mhd::EmptyBoundaryCondition<Real> bc;
  HeatingSource src;

  explicit Model(Config &config)
      : mhd::ModelBase<Model<Backend>, Real, Backend>(config), eos(config),
        ic{eos}, bc(), src() {}
};

/// @brief Advance one step with `dt` and check ei = ei0 + dt * heating / ro
template <typename Backend> void check_heating(Real dt) {
  Config config(std::string(CONFIG_DIR) + "config_source.yaml");
  Model<Backend> model(config);
  auto &mhd = model.mhd;
  mhd.apply_initial_condition(model.ic, model.bc);
  mhd.update(dt, model.eos, model.bc, model.src);

  const auto &g = model.grid;
  mhd::Fields<Real, backend::Host> qq(g);
  qq.copy_from(mhd.qq);
  const Real ei0 = pr0 / (model.eos.gm - 1.0) / ro0;
  for (int i = g.i_margin; i < g.i_total - g.i_margin; ++i) {
    for (int j = g.j_margin; j < g.j_total - g.j_margin; ++j) {
      for (int k = g.k_margin; k < g.k_total - g.k_margin; ++k) {
        CHECK(qq.ro(i, j, k) == doctest::Approx(ro0).epsilon(1e-14));
        CHECK(qq.ei(i, j, k) ==
              doctest::Approx(ei0 + dt * heating / ro0).epsilon(1e-13));
      }
    }
  }
}

}  // namespace test_source_update
