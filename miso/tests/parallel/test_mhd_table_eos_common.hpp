#pragma once

#include <cmath>
#include <string>

#include <doctest/doctest.h>

#include <miso/core.hpp>
#include <miso/eos_table.hpp>
#include <miso/mhd.hpp>

// Integration test of eos::TableEOS through the MHD solver.
//
// The state is uniform (v = 0, B = 0) at a table point in the photosphere, with
// periodic boundaries. The time step from the CFL condition is then
// cfl_number * min(dx) / c_s with the sound speed of the table, and one step
// with a constant heating rate q gives e = e0 + dt q / rho. `eos.gm` in
// config_source.yaml is not used by TableEOS.
//
// The pressure from the table is checked with a small sinusoidal perturbation
// e = e1 (1 + eps sin kx) at rest: after one short step, rho vx = -dt dp/dx
// with dp/dx = (dp/de) e1 eps k cos kx, where dp/de is taken from the table.

namespace test_mhd_table_eos {

using Real = double;
using namespace miso;

// A table point in input_data/eos_table_roen_blend.tbl (see test_eos_table.cpp)
const Real ro0 = std::exp(-15.377656523015137);
const Real ei0 = std::exp(27.981135057266535);
constexpr Real cs0 = 463724.90858549863;
constexpr Real heating = 2.5e3;

struct UniformState {
  void apply(mhd::FieldsView<Real> qq, GridView<const Real>) const {
    for (int n = 0; n < qq.size(); ++n) {
      qq.ro[n] = ro0;
      qq.vx[n] = qq.vy[n] = qq.vz[n] = 0.0;
      qq.bx[n] = qq.by[n] = qq.bz[n] = 0.0;
      qq.ei[n] = ei0;
      qq.ph[n] = 0.0;
    }
  }
};

struct HeatingSource {
  __host__ __device__ Real ei(mhd::FieldsView<const Real>, int, int, int) const {
    return heating;
  }
};

// Between two table points in ln e, so that p is smooth in e around e1
const Real ei1 = std::exp(27.981135057266535 + 0.015);
constexpr Real eps = 1e-4;
constexpr Real two_pi = 6.283185307179586;

struct PerturbedState {
  void apply(mhd::FieldsView<Real> qq, GridView<const Real> grid) const {
    for (int i = 0; i < qq.extent(0); ++i) {
      for (int j = 0; j < qq.extent(1); ++j) {
        for (int k = 0; k < qq.extent(2); ++k) {
          qq.ro(i, j, k) = ro0;
          qq.vx(i, j, k) = qq.vy(i, j, k) = qq.vz(i, j, k) = 0.0;
          qq.bx(i, j, k) = qq.by(i, j, k) = qq.bz(i, j, k) = 0.0;
          qq.ei(i, j, k) = ei1 * (1.0 + eps * std::sin(two_pi * grid.x[i]));
          qq.ph(i, j, k) = 0.0;
        }
      }
    }
  }
};

template <typename Backend, typename InitialCondition = UniformState>
struct Model
    : public mhd::ModelBase<Model<Backend, InitialCondition>, Real, Backend> {
  eos::TableEOS<Real, Backend> eos;
  InitialCondition ic;
  mhd::EmptyBoundaryCondition<Real> bc;
  HeatingSource src;

  explicit Model(Config &config)
      : mhd::ModelBase<Model<Backend, InitialCondition>, Real, Backend>(config),
        eos(fs::path(INPUT_DATA_DIR) / "eos_table_roen_blend.tbl") {}
};

template <typename Backend> void check_table_eos() {
  Config config(std::string(CONFIG_DIR) + "config_source.yaml");
  Model<Backend> model(config);
  model.mhd.apply_initial_condition(model.ic, model.bc);

  const Real dt = model.mhd.cfl(model.eos);
  const Real cfl_number = config["mhd"]["cfl_number"].as<Real>();
  CHECK(dt ==
        doctest::Approx(cfl_number * model.grid.min_dxyz / cs0).epsilon(1e-10));

  model.mhd.update(dt, model.eos, model.bc, model.src);
  const auto &g = model.grid;
  mhd::Fields<Real, backend::Host> qq(g);
  qq.copy_from(model.mhd.qq);
  const Real ei = ei0 + dt * heating / ro0;
  for (int i = g.i_margin; i < g.i_total - g.i_margin; ++i) {
    for (int j = g.j_margin; j < g.j_total - g.j_margin; ++j) {
      for (int k = g.k_margin; k < g.k_total - g.k_margin; ++k) {
        CHECK(qq.ro(i, j, k) == doctest::Approx(ro0).epsilon(1e-12));
        CHECK(std::abs(qq.vx(i, j, k)) < 1e-12 * cs0);  // round-off
        CHECK(qq.ei(i, j, k) == doctest::Approx(ei).epsilon(1e-12));
      }
    }
  }
}

template <typename Backend> void check_pressure_gradient() {
  Config config(std::string(CONFIG_DIR) + "config_table_eos.yaml");
  Model<Backend, PerturbedState> model(config);
  model.mhd.apply_initial_condition(model.ic, model.bc);

  // dp/de at (ro0, ei1) directly from the arrays in the table file (not via
  // TableEOS). ro0 is on a table point, and ln p is linear in ln e between
  // the two table points around ei1.
  const eos::TableFile file(fs::path(INPUT_DATA_DIR) /
                            "eos_table_roen_blend.tbl");
  const int n_ro = file.shapes.at("ro")[0], n_en = file.shapes.at("en")[0];
  const auto &log_ro = file.get("ro", {n_ro});
  const auto &log_en = file.get("en", {n_en});
  const auto &pr = file.get("pr", {n_ro, n_en});
  int i_ro = 0, j_en = 0;
  while (i_ro < n_ro - 1 && std::abs(log_ro[i_ro] - std::log(ro0)) > 1e-10) {
    ++i_ro;
  }
  while (j_en < n_en - 2 && log_en[j_en + 1] < std::log(ei1)) {
    ++j_en;
  }
  REQUIRE(std::abs(log_ro[i_ro] - std::log(ro0)) < 1e-10);
  REQUIRE(log_en[j_en] < std::log(ei1));
  REQUIRE(std::log(ei1) < log_en[j_en + 1]);
  const Real slope =
      std::log(pr[i_ro * n_en + j_en + 1] / pr[i_ro * n_en + j_en]) /
      (log_en[j_en + 1] - log_en[j_en]);
  const Real pr1 =
      pr[i_ro * n_en + j_en] * std::exp(slope * (std::log(ei1) - log_en[j_en]));
  const Real dpde = slope * pr1 / ei1;

  // A short step, so that the O(dt^2) terms are negligible
  const Real dt = 0.1 * model.mhd.cfl(model.eos);
  model.mhd.update(dt, model.eos, model.bc, model.src);
  const auto &g = model.grid;
  mhd::Fields<Real, backend::Host> qq(g);
  qq.copy_from(model.mhd.qq);
  const Real amplitude = dt * dpde * ei1 * eps * two_pi;
  for (int i = g.i_margin; i < g.i_total - g.i_margin; ++i) {
    const Real expected = -amplitude * std::cos(two_pi * g.x[i]);
    CHECK(ro0 * qq.vx(i, g.j_margin, g.k_margin) ==
          doctest::Approx(expected).scale(amplitude).epsilon(1e-2));
  }
}

}  // namespace test_mhd_table_eos
