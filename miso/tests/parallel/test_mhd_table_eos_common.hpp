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
// with a constant heating rate q gives e = e0 + dt q / rho.

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

template <typename Backend>
struct Model : public mhd::ModelBase<Model<Backend>, Real, Backend> {
  eos::TableEOS<Real, Backend> eos;
  UniformState ic;
  mhd::EmptyBoundaryCondition<Real> bc;
  HeatingSource src;

  explicit Model(Config &config)
      : mhd::ModelBase<Model<Backend>, Real, Backend>(config),
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

}  // namespace test_mhd_table_eos
