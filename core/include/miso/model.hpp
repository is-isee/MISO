#pragma once

#include <cmath>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <string>
#include <vector>

#include <miso/cuda_compat.hpp>
#include <miso/eos.hpp>
#include <miso/grid.hpp>
#include <miso/mhd.hpp>
#include <miso/mpi_manager.hpp>
#include <miso/radiative_transfer.hpp>
#include <miso/time.hpp>
#ifdef USE_CUDA
#include <miso/cuda_utils.cuh>
#endif

namespace miso {

template <typename Real> struct Model {
  Config &config;
  MPITopology mpi;
  Time<Real> time;
  Grid<Real> grid_global;
  Grid<Real> grid_local;
  EOS<Real> eos;
  mhd::MHD<Real> mhd;

#ifdef USE_CUDA
  GridDevice<Real> grid_d;
  CudaKernelShape<Real> cu_shape;
  mhd::MHDStreams mhd_streams;
  mhd::MHDDevice<Real> mhd_d;
#endif

  Model(Config &config_)
      : config(config_), mpi(config_), time(config.yaml_obj),
        grid_global(config.yaml_obj), grid_local(grid_global, mpi), eos(config),
#ifdef USE_CUDA
        mhd(grid_local), mhd_d(grid_local, mhd), grid_d(grid_local),
        cu_shape(grid_local)
#else
        mhd(grid_local)
#endif
  {
  }

  void save_metadata() {
    config.create_save_directory();
    MPI_Barrier(mpi.cart_comm);
    config.save();
    grid_global.save(config);
    mpi.save(config.mpi_save_dir);
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

      if (mpi.myrank == 0) {
        std::cout << std::fixed << std::setprecision(2)
                  << "time = " << std::setw(6) << time.time
                  << ";  n_step = " << std::setw(8) << time.n_step
                  << ";  n_output = " << std::setw(8) << time.n_output
                  << std::endl;
      }

      time.n_output++;
    }
  }
};

}  // namespace miso
