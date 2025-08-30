#pragma once

#include <cmath>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <string>
#include <vector>

#include "eos.hpp"
#include "grid_cpu.hpp"
#include "mpi_manager.hpp"
#include "time_cpu.hpp"

#include "mhd_cpu.hpp"
#include "radiative_transfer_cpu.hpp"
#ifdef USE_CUDA
#include "cuda_manager.cuh"
#include "grid_gpu.cuh"
#include "mhd_gpu.cuh"
#include "time_gpu.cuh"
#endif

template <typename Real> struct Model {
  Config &config;
  MPIManager<Real> &mpi;
  Time<Real> time;
  Grid<Real> grid_global;
  Grid<Real> grid_local;
  EOS<Real> eos;
  MHD<Real> mhd;

  /// TODO: make num_rays configurable
  static constexpr int num_rays = 24;
  RT<Real> rt;

#ifdef USE_CUDA
  GridDevice<Real> grid_d;
  MHDDevice<Real> mhd_d;
  CudaManager<Real> cuda;
  TimeDevice<Real> time_d;
#endif

  Model(Config &config_)
      : config(config_), mpi(config_.mpi), time(config.yaml_obj),
        grid_global(config.yaml_obj), grid_local(grid_global, mpi), eos(config),
#ifdef USE_CUDA
        mhd_d(grid_local, mhd), grid_d(grid_local), cuda(grid_local),
        time_d(cuda),
#endif
        mhd(grid_local), rt(grid_local, num_rays) {
  }

  Model(Config &config_, Time<Real> &time_, Grid<Real> &grid_global_,
        Grid<Real> &grid_local_, EOS<Real> &eos_, MHD<Real> &mhd_, RT<Real> &rt_,
        MPIManager<Real> &mpi_)
      : config(config_), time(time_), grid_global(grid_global_),
        grid_local(grid_local_), eos(eos_), mpi(mpi_),
#ifdef USE_CUDA
        mhd_d(grid_local, mhd_), grid_d(grid_local), cuda(grid_local),
        time_d(cuda),
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

    if (this->time.time >= this->time.dt_output * this->time.n_output) {
#ifdef USE_CUDA
      this->mhd_d.qq.copy_to_host(this->mhd.qq, this->cuda);
#endif
      this->save_state();

      if (this->mpi.myrank == 0) {
        std::cout << std::fixed << std::setprecision(2)
                  << "time = " << std::setw(6) << this->time.time
                  << ";  n_step = " << std::setw(8) << this->time.n_step
                  << ";  n_output = " << std::setw(8) << this->time.n_output
                  << std::endl;
      }

      time.n_output++;
    }
  }
};
