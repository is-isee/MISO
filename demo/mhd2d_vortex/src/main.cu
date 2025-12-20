#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <miso/config.hpp>
#include <miso/model.hpp>
#include <miso/mpi_manager.hpp>
#include <miso/time_integrator_gpu.cuh>
#include <miso/types.hpp>
#include <miso/utility.hpp>

using miso::Real, miso::pi;

void initial_condition(miso::Model<Real> &model) {
  miso::mhd::MHDCore<Real> &qq = model.mhd.qq;
  const auto &grid = model.grid_local;
  const auto &eos = model.eos;

  Real b0 = std::sqrt(4.0 * pi<Real>) / eos.gm;
  Real v0 = 1.0;

  for (int i = 0; i < grid.i_total; ++i) {
    for (int j = 0; j < grid.j_total; ++j) {
      for (int k = 0; k < grid.k_total; ++k) {
        qq.ro(i, j, k) = 1.0;
        Real pr = 1.0 / eos.gm;
        qq.ei(i, j, k) = pr / (eos.gm - 1.0) / qq.ro(i, j, k);
        qq.vx(i, j, k) = -v0 * std::sin(2.0 * pi<Real> * grid.y[j]);
        qq.vy(i, j, k) = +v0 * std::sin(2.0 * pi<Real> * grid.x[i]);
        qq.vz(i, j, k) = 0.0;
        qq.bx(i, j, k) = -b0 * std::sin(2.0 * pi<Real> * grid.y[j]);
        qq.by(i, j, k) = +b0 * std::sin(4.0 * pi<Real> * grid.x[i]);
        qq.bz(i, j, k) = 0.0;
      }
    }
  }
}

int main() {
  std::string config_dir = CONFIG_DIR;

  miso::MPIManager mpi;
  miso::Config config(config_dir + "config.yaml", mpi);
  mpi.setup_mpi(config.yaml_obj);
  miso::Model<Real> model(config);
  model.save_metadata();

  initial_condition(model);

  miso::mhd::TimeIntegrator<Real> time_integrator(model);
  time_integrator.run();

  return 0;
}
