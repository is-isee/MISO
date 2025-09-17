#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "config.hpp"
#include "model.hpp"
#include "mpi_manager.hpp"
#include "types.hpp"
#include "utility.hpp"

#include "force.hpp"

#ifdef USE_CUDA
#include "time_integrator_gpu.cuh"
#else
#include "time_integrator_cpu.hpp"
#endif

template <typename Real> void initial_condition(Model<Real> &model);

int main() {
  std::string config_dir = CONFIG_DIR;

  MPIManager<Real> mpi;
  Config config(config_dir + "config.yaml", mpi);
  mpi.setup_mpi(config.yaml_obj);
  Model<Real> model(config);
  model.save_metadata();

  initial_condition<Real>(model);

  TimeIntegrator<Real> time_integrator(model);
  time_integrator.run();

  return 0;
}

template <typename Real> void initial_condition(Model<Real> &model) {
  MHDCore<Real> &qq = model.mhd.qq;
  const auto &grid = model.grid_local;
  const auto &eos = model.eos;

  Real v_amp = 1.e-3;

  std::mt19937 engine(model.mpi.myrank);
  std::uniform_real_distribution<Real> dist(-1.0, 1.0);

  for (int i = 0; i < grid.i_total; ++i) {
    for (int j = 0; j < grid.j_total; ++j) {
      for (int k = 0; k < grid.k_total; ++k) {

        // initial condition
        if (grid.y[j] > 0.0) {
          qq.ro(i, j, k) = 2.0;
        } else {
          qq.ro(i, j, k) = 1.0;
        }
        Real pr0 = 2.5;
        Real pr = pr0 - g_grav * qq.ro(i, j, k) * grid.y[j];

        qq.ei(i, j, k) = pr / (eos.gm - 1.0) / qq.ro(i, j, k);
        qq.vx(i, j, k) = qq.vx(i, j, k) + v_amp * dist(engine);
        qq.vy(i, j, k) = v_amp * dist(engine);
        qq.vz(i, j, k) = 0.0;

        qq.bx(i, j, k) = 0.0;
        qq.by(i, j, k) = 0.0;
        qq.bz(i, j, k) = 0.0;
      }
    }
  }
}
