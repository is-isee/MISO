#pragma once

#include <doctest/doctest.h>
#include <filesystem>

#include <miso/config.hpp>
#include <miso/grid.hpp>
#include <miso/model.hpp>
#include <miso/types.hpp>

// Test the Model constructor and accessors
inline miso::Model<miso::Real> run_test_model() {
  using namespace miso;

  // assuming current path is build/tests
  MPIManager mpi;
  std::string config_dir = CONFIG_DIR;

  Config config(config_dir + "config_model.yaml", mpi);
  mpi.setup_mpi(config.yaml_obj);
  Time<Real> time(config.yaml_obj);
  Grid<Real> grid_global(config.yaml_obj);

  Grid<Real> grid_local(grid_global, mpi);
  EOS<Real> eos(config);
  mhd::MHD<Real> mhd(grid_local);
  rt::RT<Real> rt(grid_local, 24);

  miso::Model<Real> model(config, time, grid_global, grid_local, eos, mhd, rt,
                          mpi);

  // Check dimensions
  REQUIRE(model.grid_global.i_size == grid_global.i_size);
  REQUIRE(model.grid_global.j_size == grid_global.j_size);
  REQUIRE(model.grid_global.k_size == grid_global.k_size);
  REQUIRE(model.grid_global.margin == grid_global.margin);
  REQUIRE(model.grid_global.i_total == grid_global.i_total);
  REQUIRE(model.grid_global.j_total == grid_global.j_total);
  REQUIRE(model.grid_global.k_total == grid_global.k_total);

  REQUIRE(model.grid_local.i_size == grid_local.i_size);
  REQUIRE(model.grid_local.j_size == grid_local.j_size);
  REQUIRE(model.grid_local.k_size == grid_local.k_size);
  REQUIRE(model.grid_local.margin == grid_local.margin);
  REQUIRE(model.grid_local.i_total == grid_local.i_total);
  REQUIRE(model.grid_local.j_total == grid_local.j_total);
  REQUIRE(model.grid_local.k_total == grid_local.k_total);

  // Check time
  REQUIRE(model.time.tend == time.tend);
  REQUIRE(model.time.dt_output == time.dt_output);
  REQUIRE(model.time.n_output_digits == time.n_output_digits);

  return model;
}
