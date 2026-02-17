#include "bc.hpp"
#include "common.hpp"
#include "ic.hpp"
#include "sources.hpp"
#include "timestep.hpp"

using EOS = eos::IdealEOS<Real>;

struct Model {
  Config &config;
  mpi::Shape mpi_shape;
  Time<Real> time;
  Grid<Real, backend::Host> grid;

  mhd::ExecContext<Backend> exec_ctx;
  EOS eos;
  mhd::MHD<Real, Backend> mhd;
  InitialCondition<EOS> ic;
  BoundaryCondition<EOS> bc;
  ExternalSources<Real> src;
  TimeStep timestep;

  Model(Config &config)
      : config(config), mpi_shape(config), time(config), grid(config, mpi_shape),
        exec_ctx(mpi_shape, grid), eos(config), mhd(config, grid, exec_ctx),
        ic(config, eos), bc(config, mpi_shape, mhd.grid, eos),
        src(config, mhd.grid), timestep(config, mhd.grid) {}

  void save_metadata() {
    MPI_Barrier(mpi::comm());
    config.save();
    grid.save(config);
    exec_ctx.mpi_shape.save();
  }

  void save_state() {
    time.save();
    mhd.save(time);
  }

  void load_state() {
    time.load();
    mhd.load(time);
  }

  void save_if_needed() {
    if (time.time < time.dt_output * time.n_output)
      return;

    save_state();
    if (mpi::is_root()) {
      std::cout << std::fixed << std::setprecision(2) << "time = " << std::setw(6)
                << time.time << ";  n_step = " << std::setw(8) << time.n_step
                << ";  n_output = " << std::setw(8) << time.n_output << std::endl;
    }
    time.n_output++;
  }

  // Main time integration loop
  void run() {
    // Apply initial condition (load if continue is true)
    mhd.apply_initial_condition(ic, bc);
    bc.qq_init.copy_from(mhd.qq);  // store initial state for boundary condition
    if (config["base"]["continue"].as<bool>() &&
        fs::exists(time.time_save_dir + "n_output.txt")) {
      load_state();
    }

    save_metadata();
    save_if_needed();

    while (time.time < time.tend) {
      // Determine time step size
      const auto dt = timestep.cfl(mhd.qq, mhd.grid, eos);

      // Update MHD variables
      mhd.update(dt, eos, bc, src);
      time.update(dt);

      save_if_needed();
    }
  }
};

int main(int argc, char *argv[]) {
  using namespace miso;

  // Initialize MPI and CUDA environments
  Env ctx(argc, argv);

  // Read configuration file
  auto config_path = parse_config_filepath(argc, argv);
  Config config(config_path.value_or("./config.yaml"));

  // Run simulation
  Model model(config);
  model.run();
}
