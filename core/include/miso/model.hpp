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
  MPIManager &mpi;
  Time<Real> time;
  Grid<Real> grid_global;
  Grid<Real> grid_local;
  EOS<Real> eos;
  mhd::MHD<Real> mhd;

  /// TODO: make num_rays configurable
  static constexpr int num_rays = 24;
  rt::RT<Real> rt;

#ifdef USE_CUDA
  GridDevice<Real> grid_d;
  CudaKernelShape<Real> cu_shape;
  mhd::MHDStreams mhd_streams;
  mhd::MHDDevice<Real> mhd_d;
#endif

  Model(Config &config_)
      : config(config_), mpi(config_.mpi), time(config.yaml_obj),
        grid_global(config.yaml_obj), grid_local(grid_global, mpi), eos(config),
#ifdef USE_CUDA
        mhd_d(grid_local, mhd), grid_d(grid_local), cu_shape(grid_local),
#endif
        mhd(grid_local), rt(grid_local, num_rays) {
  }

  Model(Config &config_, Time<Real> &time_, Grid<Real> &grid_global_,
        Grid<Real> &grid_local_, EOS<Real> &eos_, mhd::MHD<Real> &mhd_,
        rt::RT<Real> &rt_, MPIManager &mpi_)
      : config(config_), time(time_), grid_global(grid_global_),
        grid_local(grid_local_), eos(eos_), mpi(mpi_),
#ifdef USE_CUDA
        mhd_d(grid_local, mhd_), grid_d(grid_local), cu_shape(grid_local),
        mhd_streams(),
#endif
        mhd(mhd_), rt(rt_) {
  }

  void save_metadata() {
    this->config.create_save_directory();
    MPI_Barrier(this->mpi.cart_comm);
    this->config.save();
    this->grid_global.save(this->config);
    this->save_mpi_coords();
  }

  void save_state() {
    this->mhd.save(this->config, this->time);
    this->time.save(this->config);
  }

  void load_state() {
    this->time.load(this->config);
    this->mhd.load(this->config, this->time);
  }

  void save_mpi_coords() {
    int all_coords[this->mpi.ndims * this->mpi.n_procs];
    MPI_Gather(this->mpi.coord, this->mpi.ndims, MPI_INT, all_coords,
               this->mpi.ndims, MPI_INT, 0, this->mpi.cart_comm);

    if (this->mpi.myrank == 0) {
      std::ofstream ofs(config.mpi_save_dir + "/coords.csv");
      ofs << "rank,x,y,z\n";
      for (int rank = 0; rank < this->mpi.n_procs; ++rank) {
        ofs << rank << "," << all_coords[rank * 3 + 0] << ","
            << all_coords[rank * 3 + 1] << "," << all_coords[rank * 3 + 2]
            << "\n";
      }
    }
  }

  void save_if_needed() {
    if (time.time >= time.dt_output * time.n_output) {
#ifdef USE_CUDA
      mhd_d.qq.copy_to_host(mhd.qq, mhd_streams);
#endif
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
