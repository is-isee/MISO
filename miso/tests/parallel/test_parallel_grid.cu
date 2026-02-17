#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <miso/config.hpp>
#include <miso/env.hpp>
#include <miso/grid.hpp>
#include <miso/mpi_util.hpp>
#include <miso/types.hpp>

using namespace miso;

TEST_CASE("Test MPI" * doctest::test_suite("mpi")) {
  Env env;
  std::string config_dir = CONFIG_DIR;
  std::vector<std::string> directions = {"x", "y", "z"};

  for (const auto &direction : directions) {
    const auto &config_path = config_dir + "config_mpi_" + direction + ".yaml";
    Config config(config_path);
    mpi::Shape mpi_shape(config);
    Grid<float, backend::Host> grid_h(config, mpi_shape);
    Grid<float, backend::CUDA> grid_d(grid_h);
    grid_h.copy_from(grid_d);
    REQUIRE(grid_h.i_size ==
            config["grid"]["i_size"].as<int>() / mpi_shape.x_procs);
    REQUIRE(grid_h.j_size ==
            config["grid"]["j_size"].as<int>() / mpi_shape.y_procs);
    REQUIRE(grid_h.k_size ==
            config["grid"]["k_size"].as<int>() / mpi_shape.z_procs);
  }
}
