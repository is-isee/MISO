#include <string>

#include <miso/config.hpp>
#include <miso/env.hpp>
#include <miso/eos.hpp>
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

void initial_condition(mhd::impl_host::Fields<Real> &qq, const Grid<Real> &grid,
                       const eos::IdealEOS<Real> &eos) {
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

// Periodic boundary condition is applied by MPI communication.
// Be sure to set "periodic" in domain field of config.yaml.
struct EmptyBC {
  explicit EmptyBC(Config &config) {}
  void apply(const mhd::FieldsView<Real> &qq) {}
};

struct Model {
  Config &config;
  mpi::Shape mpi_shape;
  Time<Real> time;
  Grid<Real> grid_global;
  Grid<Real> grid;

  eos::IdealEOS<Real> eos;
#ifdef USE_CUDA
  cuda::KernelShape3D cu_shape;
  mhd::impl_cuda::Streams mhd_streams;
  mhd::impl_cuda::ExecContext exec_ctx;
  mhd::MHD<Real, EmptyBC, eos::IdealEOS<Real>, mhd::impl_cuda::NoSource<Real>>
      mhd;
#else
  mhd::impl_host::ExecContext exec_ctx;
  mhd::MHD<Real, EmptyBC, eos::IdealEOS<Real>, mhd::impl_host::NoSource<Real>>
      mhd;
#endif

  Model(Config &config)
      : config(config), mpi_shape(config), time(config), grid_global(config),
        grid(grid_global, mpi_shape), eos(config),
#ifdef USE_CUDA
        cu_shape(grid), mhd_streams(), exec_ctx{mpi_shape, cu_shape, mhd_streams},
        mhd(config, time, grid, exec_ctx)
#else
        exec_ctx{mpi_shape}, mhd(config, time, grid, exec_ctx)
#endif
  {
  }

  void save_metadata() {
    MPI_Barrier(mpi::comm());
    config.save();
    grid_global.save(config);
    mpi_shape.save();
  }

  void save_state() {
#ifdef USE_CUDA
    mhd.qq_d.copy_to_host(mhd.qq, mhd_streams);
#endif
    mhd.save();
    time.save();
  }

  void load_state() {
    time.load();
    mhd.load();
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
    if (config["base"]["continue"].template as<bool>() &&
        fs::exists(config.time_save_dir + "n_output.txt")) {
      load_state();
    }

    save_metadata();
    initial_condition(mhd.qq, grid, eos);
#ifdef USE_CUDA
    mhd.qq_d.copy_from_host(mhd.qq, mhd_streams);
#endif

    MPI_Barrier(mpi::comm());

    save_if_needed();
    while (time.time < time.tend) {
      // basic MHD time integration
      const auto dt = mhd.cfl();
      mhd.update(dt);

      // Time is update after all procedures
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
