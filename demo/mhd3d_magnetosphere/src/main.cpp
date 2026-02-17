#include "common.hpp"

#include "bc.hpp"
#include "ic.hpp"
#include "src.hpp"
#include "timestep.hpp"

struct Model : public mhd::ModelBase<Model, Real, Backend> {
  eos::IdealEOS<Real> eos;
  InitialCondition ic;
  BoundaryCondition bc;
  SourceTerm src;
  TimeStep timestep;

  void update() {
    const auto dt = timestep.cfl(mhd.qq, mhd.grid, eos);
    mhd.update(dt, eos, bc, src);
    time.update(dt);
  }

  Model(Config &config)
      : ModelBase(config), eos(config), ic(config, eos),
        bc(config, mpi_shape, mhd.grid, eos), src(config, mhd.grid),
        timestep(config, mhd.grid) {}
};

int main(int argc, char *argv[]) {
  using namespace miso;

  // Initialize MPI and CUDA environments
  Env env(argc, argv);

  // Read configuration file
  auto config_path = parse_config_filepath(argc, argv);
  Config config(config_path.value_or("./config.yaml"));

  // Run simulation
  Model model(config);
  model.run();
}
