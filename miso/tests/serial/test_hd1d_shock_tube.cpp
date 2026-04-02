#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <cmath>
#include <doctest/doctest.h>
#include <hd1d_boundary_condition.hpp>
#include <hd1d_initial_condition.hpp>
#include <istream>
#include <sod_solution.hpp>

#include <miso/boundary_condition.hpp>
#include <miso/mhd_model_base.hpp>

using namespace miso;
using Real = float;

#ifdef USE_CUDA
using Backend = backend::CUDA;
#else
using Backend = backend::Host;
#endif

// This test use demo/hd1d_shock_tube and validate the numerical solution by comparing with the exact solution of Sod's shock tube problem.
// The test runs the simulation for three different configurations (shock tube in x, y, and z directions) and checks the consistency of the results among them as well as their agreement with the exact solution.

struct Model : public mhd::ModelBase<Model, Real, Backend> {
  eos::IdealEOS<Real> eos;
  InitialCondition<Real> ic;
  BoundaryCondition<Real> bc;
  mhd::EmptySourceTerm<Real> src;

  Model(Config &config)
      : ModelBase(config), eos(config), ic(config, eos), bc(mpi_shape), src() {}
};

TEST_CASE("Test HD 1D Shock Tube" * doctest::test_suite("hd1d_shock_tube")) {
  Env env;

  Config config_x("../../../../demo/hd1d_shock_tube/config/config_x.yaml");
  Model model_x(config_x);
  model_x.run();

  Config config_y("../../../../demo/hd1d_shock_tube/config/config_y.yaml");
  Model model_y(config_y);
  model_y.run();

  Config config_z("../../../../demo/hd1d_shock_tube/config/config_z.yaml");
  Model model_z(config_z);
  model_z.run();

  std::cout << model_x.ic.prl << std::endl;

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

  for (std::size_t i = 0; i < model_x.grid.i_total; ++i) {
    CHECK(model_x.mhd.qq.ro[i] == doctest::Approx(model_y.mhd.qq.ro[i]));
    CHECK(model_x.mhd.qq.ro[i] == doctest::Approx(model_z.mhd.qq.ro[i]));
  }
}
