#include <hd1d_boundary_condition.hpp>
#include <hd1d_initial_condition.hpp>
#include <miso/boundary_condition.hpp>
#include <miso/mhd_model_base.hpp>

using namespace miso;

using Real = float;

#ifdef USE_CUDA
using Backend = backend::CUDA;
#else
using Backend = backend::Host;
#endif

struct Model : public mhd::ModelBase<Model, Real, Backend> {
  eos::IdealEOS<Real> eos;
  // defined in hd1d_initial_condition.hpp
  InitialCondition<Real> ic;
  // defined in hd1d_boundary_condition.hpp
  BoundaryCondition<Real, Backend> bc;
  mhd::EmptySourceTerm<Real> src;

  Model(Config &config)
      : ModelBase(config), eos(config), ic(config, eos), bc(mpi_shape), src() {}
};

int main(int argc, char **argv) {
  // Initialize MPI and CUDA environments
  Env env(argc, argv);

  // Read configuration file
  auto config_path = parse_config_filepath(argc, argv);
  // by default, x-direction shock tube config is loaded.
  Config config(config_path.value_or("./config/config_x.yaml"));

  // Run simulation
  Model model(config);
  model.run();
}
