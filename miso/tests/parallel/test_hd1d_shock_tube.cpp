#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <cmath>
#include <doctest/doctest.h>
#include <filesystem>
#include <test_hd1d_shock_tube/boundary_condition.hpp>
#include <test_hd1d_shock_tube/initial_condition.hpp>

#include <test_hd1d_shock_tube/sod_solution.hpp>

#include <miso/boundary_condition.hpp>
#include <miso/mhd_model_base.hpp>

using namespace miso;
using Real = float;

using Backend = backend::Host;

// This test validates the numerical solution by comparing with the exact solution of Sod's shock tube problem.
// The test runs the simulation for three different configurations (shock tube in x, y, and z directions) and checks the consistency of the results among them as well as their agreement with the exact solution.

struct Model : public mhd::ModelBase<Model, Real, Backend> {
  eos::IdealEOS<Real> eos;
  InitialCondition<Real> ic;
  BoundaryCondition<Real, Backend> bc;
  mhd::EmptySourceTerm<Real> src;

  Model(Config &config)
      : ModelBase(config), eos(config), ic(config, eos), bc(mpi_shape), src() {}
};

TEST_CASE("Test HD 1D Shock Tube" * doctest::test_suite("hd1d_shock_tube")) {
  Env env;

  std::filesystem::path config_dir =
      std::filesystem::path(HD1D_SHOCK_TUBE_CONFIG_DIR);
  std::filesystem::path config_x_path = config_dir / "config_x.yaml";
  std::filesystem::path config_y_path = config_dir / "config_y.yaml";
  std::filesystem::path config_z_path = config_dir / "config_z.yaml";

  Config config_x(config_x_path.string());
  Model model_x(config_x);
  model_x.run();

  Config config_y(config_y_path.string());
  Model model_y(config_y);
  model_y.run();

  Config config_z(config_z_path.string());
  Model model_z(config_z);
  model_z.run();

  Real csr = std::sqrt(model_x.eos.gm * model_x.ic.prr / model_x.ic.ror);
  Real csl = std::sqrt(model_x.eos.gm * model_x.ic.prl / model_x.ic.rol);

  Real xm = 0.5 * (model_x.grid.x_min + model_x.grid.x_max);

  SodSolution<Real> ss(model_x.grid.x);
  ss.calc_sod_solution(model_x.time.time, model_x.eos.gm, xm, csl, csr,
                       model_x.ic.ror, model_x.ic.rol, model_x.ic.prr,
                       model_x.ic.prl, model_x.ic.vvl, model_x.ic.vvr);

  Real ro_diff_sum = 0.0, vx_diff_sum = 0.0;
  Real ro_sum = 0.0, vx_sum = 0.0;
  for (std::size_t i = 0; i < model_x.grid.i_total; ++i) {
    ro_diff_sum += std::abs(model_x.mhd.qq.ro[i] - ss.ro[i]);
    vx_diff_sum += std::abs(model_x.mhd.qq.vx[i] - ss.vx[i]);
    ro_sum += model_x.mhd.qq.ro[i];
    vx_sum += model_x.mhd.qq.vx[i];
  }
  CHECK(ro_diff_sum / ro_sum < 5.e-2);
  CHECK(vx_diff_sum / vx_sum < 5.e-2);

  constexpr double single_precision_epsilon = 1e-5;
  for (std::size_t i = 0; i < model_x.grid.i_total; ++i) {
    CHECK(model_x.mhd.qq.ro[i] == doctest::Approx(model_y.mhd.qq.ro[i])
                                      .epsilon(single_precision_epsilon)
                                      .scale(1.0));
    CHECK(model_x.mhd.qq.ro[i] == doctest::Approx(model_z.mhd.qq.ro[i])
                                      .epsilon(single_precision_epsilon)
                                      .scale(1.0));
  }
}
