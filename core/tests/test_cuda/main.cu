#ifdef USE_CUDA
#include "config.hpp"
#include "model.hpp"
#include "types.hpp"
#include "utility.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "cuda_manager.cuh"
#include "cuda_runtime.h"
#include "time_integrator_gpu.cuh"

__global__ void initialize_density(MHDDevice<Real> mhd, int i_total, int j_total,
                                   int k_total) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < i_total && j < j_total && k < k_total) {
    int idx = (i * j_total + j) * k_total + k;
    mhd.qq.ro[idx] = mhd.qq.ro[idx];
  }
}

int main() {
  std::vector<std::string> directions = {"x", "y"};

  std::string config_dir = CONFIG_DIR;
  MPIManager mpi;

  for (const auto &direction : directions) {
    {
      Config config(config_dir + "config_" + direction + ".yaml", mpi);
      mpi.setup_mpi(config.yaml_obj);
      // cudaSetDevice(mpi.myrank);
      Model<Real> model(config);
      model.save_metadata();

      for (int i = 0; i < model.grid_local.i_total; ++i) {
        for (int j = 0; j < model.grid_local.j_total; ++j) {
          for (int k = 0; k < model.grid_local.k_total; ++k) {
            model.mhd.qq.ro(i, j, k) = mpi.myrank;
          }
        }
      }

      MHDDevice<Real> mhd_d(model.grid_local, model.mhd);
      model.mhd_d.qq.copy_from_host(model.mhd.qq, model.cuda);

      model.mhd_d.mpi_exchange_halo(model.mhd_d.qq, model.grid_d, model.mpi,
                                    model.cuda);
      initialize_density<<<model.cuda.grid_dim, model.cuda.block_dim>>>(
          model.mhd_d, model.grid_local.i_total, model.grid_local.j_total,
          model.grid_local.k_total);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

      cudaDeviceSynchronize();
      model.mhd_d.qq.copy_to_host(model.mhd.qq, model.cuda);

      for (int i = 0; i < model.grid_local.i_total; ++i) {
        for (int j = 0; j < model.grid_local.j_total; ++j) {
          for (int k = 0; k < model.grid_local.k_total; ++k) {
            if (direction == "x") {
              if (mpi.myrank == 0 &&
                  i >= model.grid_local.i_total - model.grid_local.i_margin) {
                assert(model.mhd.qq.ro(i, j, k) == 1);
              }
              if (mpi.myrank == 1 && i < model.grid_local.i_margin) {
                assert(model.mhd.qq.ro(i, j, k) == 0);
              }
            } else if (direction == "y") {
              if (mpi.myrank == 0 &&
                  j >= model.grid_local.j_total - model.grid_local.j_margin) {
                assert(model.mhd.qq.ro(i, j, k) == 1);
              }
              if (mpi.myrank == 1 && j < model.grid_local.j_margin) {
                assert(model.mhd.qq.ro(i, j, k) == 0);
              }
            } else if (direction == "z") {
              if (mpi.myrank == 0 &&
                  k >= model.grid_local.k_total - model.grid_local.k_margin) {
                assert(model.mhd.qq.ro(i, j, k) == 1);
              }
              if (mpi.myrank == 1 && k < model.grid_local.k_margin) {
                assert(model.mhd.qq.ro(i, j, k) == 0);
              }
            }
          }
        }
      }

      model.mhd_d.qq.free();
    }
  }
  return 0;
}
#endif
