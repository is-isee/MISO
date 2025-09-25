#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "config.hpp"
#include "custom_boundary_condition_impl.hpp"
#include "initial_condition.hpp"
#include "model.hpp"
#include "mpi_manager.hpp"
#include "time_integrator_gpu.cuh"
#include "types.hpp"
#include "utility.hpp"

int main() {
  std::string config_dir = CONFIG_DIR;

  MPIManager mpi;
  Config config(config_dir + "config.yaml", mpi);
  mpi.setup_mpi(config.yaml_obj);
  Model<Real> model(config);
  model.save_metadata();

  InitialCondition<Real> initial_condition(model);
  initial_condition.apply(model.mhd.qq);

  TimeIntegrator<Real> time_integrator(model);
  time_integrator.run();

  return 0;
}
