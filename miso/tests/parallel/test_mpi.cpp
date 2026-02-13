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
    Grid<Real, backend::Host> grid(config, mpi_shape);
    REQUIRE(grid.i_size ==
            config["grid"]["i_size"].as<int>() / mpi_shape.x_procs);
    REQUIRE(grid.j_size ==
            config["grid"]["j_size"].as<int>() / mpi_shape.y_procs);
    REQUIRE(grid.k_size ==
            config["grid"]["k_size"].as<int>() / mpi_shape.z_procs);
  }
}
