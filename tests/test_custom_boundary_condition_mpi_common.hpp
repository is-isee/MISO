#pragma once
#include <doctest/doctest.h>

#include "config.hpp"
#include "model.hpp"
#include "mpi_manager.hpp"
#ifdef USE_CUDA
#include "time_integrator_gpu.cuh"
#else
#include "time_integrator_cpu.hpp"
#endif

#include "types.hpp"

inline void run_custom_boundary_condition_mpi_tests() {
  MPIManager<Real> mpi;
  const std::string config_dir = CONFIG_DIR;
  Config config(config_dir + "config_custom_boundary_condition.yaml", mpi);
  mpi.setup_mpi(config.yaml_obj);
  Model<Real> model(config);

  TimeIntegrator<Real> time_integrator(model);

  for (int i = 0; i < model.grid_local.i_total; ++i) {
    for (int j = 0; j < model.grid_local.j_total; ++j) {
      for (int k = 0; k < model.grid_local.k_total; ++k) {
        model.mhd.qq.ro(i, j, k) = i;
      }
    }
  }

#ifdef USE_CUDA
  model.mhd_d.qq.copy_from_host(model.mhd.qq, model.cuda);
  time_integrator.bc->apply(model.mhd_d.qq);
  model.mhd_d.qq.copy_to_host(model.mhd.qq, model.cuda);
#else
  time_integrator.bc->apply(model.mhd.qq);
#endif

  for (int j = 0; j < model.grid_local.j_total; ++j) {
    for (int k = 0; k < model.grid_local.k_total; ++k) {
      REQUIRE(model.mhd.qq.ro(0, j, k) == doctest::Approx(3.0));
      REQUIRE(model.mhd.qq.ro(1, j, k) == doctest::Approx(2.0));

      REQUIRE(model.mhd.qq.ro(model.grid_local.i_total - 1, j, k) ==
              doctest::Approx(-(model.grid_local.i_total - 4)));
      REQUIRE(model.mhd.qq.ro(model.grid_local.i_total - 2, j, k) ==
              doctest::Approx(-(model.grid_local.i_total - 3)));
    }
  }
}
