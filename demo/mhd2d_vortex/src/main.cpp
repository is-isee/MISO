#include <string>

#include <miso/config.hpp>
#include <miso/env.hpp>
#include <miso/eos.hpp>
#include <miso/execution.hpp>
#include <miso/grid.hpp>
#include <miso/mhd.hpp>
#include <miso/mpi_util.hpp>
#include <miso/time.hpp>
#include <miso/types.hpp>
#include <miso/utility.hpp>
#ifdef USE_CUDA
#include <miso/cuda_util.cuh>
#endif

using namespace miso;

#ifdef USE_CUDA
using Backend = backend::CUDA;
#else
using Backend = backend::Host;
#endif

struct InitialCondition {
  // The signature must not be changed as it is called inside miso::mhd::MHD.
  template <typename EOS>
  void apply(mhd::FieldsView<Real> qq, GridView<const Real> grid,
             const EOS &eos) const {
    const Real pr = 1.0 / eos.gm;
    const Real b0 = util::sqrt(4.0 * pi<Real>) / eos.gm;
    const Real v0 = 1.0;
    Range3D range{{0, grid.i_total}, {0, grid.j_total}, {0, grid.k_total}};

    for_each(
        Backend{}, range, MISO_LAMBDA(int i, int j, int k) {
          qq.ro(i, j, k) = 1.0;
          qq.ei(i, j, k) = eos.roprtoei(qq.ro(i, j, k), pr);
          qq.vx(i, j, k) = -v0 * util::sin(2.0 * pi<Real> * grid.y[j]);
          qq.vy(i, j, k) = +v0 * util::sin(2.0 * pi<Real> * grid.x[i]);
          qq.vz(i, j, k) = 0.0;
          qq.bx(i, j, k) = -b0 * util::sin(2.0 * pi<Real> * grid.y[j]);
          qq.by(i, j, k) = +b0 * util::sin(4.0 * pi<Real> * grid.x[i]);
          qq.bz(i, j, k) = 0.0;
        });
  }
};

struct BoundaryCondition {
  // The signature must not be changed as it is called inside miso::mhd::MHD.
  template <typename EOS>
  void apply(mhd::FieldsView<Real> qq, GridView<const Real> grid,
             const EOS &eos) const {
    // Periodic boundary condition is applied by MPI communication.
    // Be sure to set "periodic" in domain field of config.yaml.
  }
};

struct Model {
  Config &config;
  mpi::Shape mpi_shape;
  Time<Real> time;
  Grid<Real, backend::Host> grid_global;
  Grid<Real, backend::Host> grid;

  mhd::ExecContext<Backend> exec_ctx;
  eos::IdealEOS<Real> eos;
  InitialCondition ic;
  BoundaryCondition bc;
  mhd::NoSource<Real> src;
  mhd::MHD<Real, eos::IdealEOS<Real>, Backend> mhd;

  Model(Config &config)
      : config(config), mpi_shape(config), time(config), grid_global(config),
        grid(grid_global, mpi_shape), exec_ctx(mpi_shape, grid), eos(config),
        mhd(config, grid, exec_ctx, eos) {}

  void save_metadata() {
    MPI_Barrier(mpi::comm());
    config.save();
    grid_global.save(config);
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
    if (time.time >= time.dt_output * time.n_output) {
      save_state();

      if (mpi::is_root()) {
        std::cout << std::fixed << std::setprecision(2)
                  << "time = " << std::setw(6) << time.time
                  << ";  n_step = " << std::setw(8) << time.n_step
                  << ";  n_output = " << std::setw(8) << time.n_output
                  << std::endl;
      }

      time.n_output++;
    }
  }

  /// @brief Main time integration loop
  void run() {
    if (config["base"]["continue"].as<bool>() &&
        fs::exists(time.time_save_dir + "n_output.txt")) {
      load_state();
    }

    save_metadata();
    mhd.apply_initial_condition(ic, bc);

    MPI_Barrier(mpi::comm());

    save_if_needed();
    while (time.time < time.tend) {
      // basic MHD time integration
      const auto dt = mhd.cfl();
      mhd.update(dt, bc, src);

      // Time is updated after all procedures
      time.update(dt);
      save_if_needed();
    }
  }
};

int main(int argc, char *argv[]) {
  using namespace miso;
  std::string config_dir = CONFIG_DIR;

  Env ctx(argc, argv);
  Config config(config_dir + "config.yaml");
  Model model(config);
  model.run();
}
