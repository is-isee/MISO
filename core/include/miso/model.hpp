#pragma once

#include <miso/cuda_compat.hpp>
#include <miso/eos.hpp>
#include <miso/grid.hpp>
#include <miso/mhd.hpp>
#include <miso/mpi_manager.hpp>
#include <miso/time.hpp>
#ifdef USE_CUDA
#include <miso/cuda_utils.cuh>
#endif

namespace miso {

template <typename Real> struct Model {
  Config &config;
  MPIManager mpi;
  Time<Real> time;
  Grid<Real> grid_global;
  Grid<Real> grid;

  EOS<Real> eos;
  mhd::MHD<Real> mhd;
#ifdef USE_CUDA
  GridDevice<Real> grid_d;
  CudaKernelShape<Real> cu_shape;
  mhd::MHDStreams mhd_streams;
  mhd::MHDDevice<Real> mhd_d;
#endif

  Model(Config &config_)
      : config(config_), mpi(config_), time(config), grid_global(config),
        grid(grid_global, mpi), eos(config),
#ifdef USE_CUDA
        mhd(grid), mhd_d(grid, mhd), grid_d(grid), cu_shape(grid)
#else
        mhd(grid)
#endif
  {
  }

  void save_metadata() {
    MPI_Barrier(mpi::comm());
    config.save();
    grid_global.save(config);
    mpi.save();
  }

  void save_state() {
#ifdef USE_CUDA
    mhd_d.qq.copy_to_host(mhd.qq, mhd_streams);
#endif
    mhd.save(config, time);
    time.save(config);
  }

  void load_state() {
    time.load(config);
    mhd.load(config, time);
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

    MPI_Barrier(mpi::comm());

    save_if_needed();
    while (time.time < time.tend) {
      // basic MHD time integration
      mhd.update();

      // Time is update after all procedures
      time.update();
      save_if_needed();
    }
  }
};

}  // namespace miso
