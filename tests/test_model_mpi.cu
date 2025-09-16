#include "test_model_mpi_common.hpp"

TEST_CASE("Test Model GPU") {
  Model<Real> model = run_test_model();
  REQUIRE(model.grid_d.i_total == model.grid_local.i_total);
  REQUIRE(model.grid_d.j_total == model.grid_local.j_total);
  REQUIRE(model.grid_d.k_total == model.grid_local.k_total);
  REQUIRE(model.grid_d.i_margin == model.grid_local.i_margin);
  REQUIRE(model.grid_d.j_margin == model.grid_local.j_margin);
  REQUIRE(model.grid_d.k_margin == model.grid_local.k_margin);
}
