
#include "config.hpp"
#include "grid_cpu.hpp"
#include "model.hpp"
#include "mpi_manager.hpp"
#include "types.hpp"
#include <doctest/doctest.h>

TEST_CASE("Test MPI" * doctest::test_suite("mpi")) {
  // Test MPI initialization and finalization
  MPIManager<Real> mpi;
  std::string config_dir = CONFIG_DIR;
  std::vector<std::string> directions = {"x", "y", "z"};

  for (const auto &direction : directions) {
    {
      Config config(config_dir + "config_mpi_" + direction + ".yaml", mpi);
      mpi.setup_mpi(config.yaml_obj);
      Model<Real> model(config);

      REQUIRE(model.grid_local.i_size == model.grid_global.i_size / mpi.x_procs);
      REQUIRE(model.grid_local.j_size == model.grid_global.j_size / mpi.y_procs);
      REQUIRE(model.grid_local.k_size == model.grid_global.k_size / mpi.z_procs);
    }
  }
}
