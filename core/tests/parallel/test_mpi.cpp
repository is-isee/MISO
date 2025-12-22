#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <miso/config.hpp>
#include <miso/grid.hpp>
#include <miso/model.hpp>
#include <miso/mpi_manager.hpp>
#include <miso/types.hpp>

TEST_CASE("Test MPI" * doctest::test_suite("mpi")) {
  // Test MPI initialization and finalization
  miso::ContextManager ctx;
  std::string config_dir = CONFIG_DIR;
  std::vector<std::string> directions = {"x", "y", "z"};

  for (const auto &direction : directions) {
    {
      const auto &config_path = config_dir + "config_mpi_" + direction + ".yaml";
      miso::Config config(config_path, ctx.mpi_env);
      miso::MPIManager mpi(config);

      miso::Model<miso::Real> model(config);
      REQUIRE(model.grid_local.i_size == model.grid_global.i_size / mpi.x_procs);
      REQUIRE(model.grid_local.j_size == model.grid_global.j_size / mpi.y_procs);
      REQUIRE(model.grid_local.k_size == model.grid_global.k_size / mpi.z_procs);
    }
  }
}
