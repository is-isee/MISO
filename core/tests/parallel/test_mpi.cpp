#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <miso/config.hpp>
#include <miso/env.hpp>
#include <miso/grid.hpp>
#include <miso/mpi_util.hpp>
#include <miso/types.hpp>

using namespace miso;

TEST_CASE("Test MPI" * doctest::test_suite("mpi")) {
  // Test MPI initialization and finalization
  Env env;
  std::string config_dir = CONFIG_DIR;
  std::vector<std::string> directions = {"x", "y", "z"};

  for (const auto &direction : directions) {
    const auto &config_path = config_dir + "config_mpi_" + direction + ".yaml";
    Config config(config_path);
    mpi::Shape mpi_shape(config);
    Grid<Real> grid_global(config);
    Grid<Real> grid_local(grid_global, mpi_shape);
    REQUIRE(grid_local.i_size == grid_global.i_size / mpi_shape.x_procs);
    REQUIRE(grid_local.j_size == grid_global.j_size / mpi_shape.y_procs);
    REQUIRE(grid_local.k_size == grid_global.k_size / mpi_shape.z_procs);
  }
}
