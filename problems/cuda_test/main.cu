#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "config.hpp"
#include "model.hpp"
#include "types.hpp"
#include "utility.hpp"

#ifdef USE_CUDA
#include "cuda_manager.cuh"
#include "time_integrator_gpu.cuh"
#else
#include "time_integrator_cpu.hpp"
#endif

__global__ void initialize_density(MHDDevice<Real> mhd, int i_total, int j_total,
                                   int k_total) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < i_total && j < j_total && k < k_total) {
    int idx = (i * j_total + j) * k_total + k;
    mhd.qq.ro[idx] = mhd.qq.ro[idx] + 1.0;
  }
}

int main() {
  std::string config_dir = CONFIG_DIR;
  MPIManager<Real> mpi;

  Config config(config_dir + "config.yaml", mpi);
  mpi.setup_mpi(config.yaml_obj);
  Model<Real> model(config);
  model.save_metadata();

  for (int i = 0; i < model.grid_local.i_total; ++i) {
    for (int j = 0; j < model.grid_local.j_total; ++j) {
      for (int k = 0; k < model.grid_local.k_total; ++k) {
        model.mhd.qq.ro(i, j, k) = 1.0;
      }
    }
  }

  // MHDDevice<Real> mhd_d;
  // mhd_d.allocate(grid);
  // MHDCore<Real> qq_h(grid.i_total, grid.j_total, grid.k_total);

  model.mhd_d.qq.copy_from_host(model.mhd.qq, cuda);

  dim3 cuda_block(8, 8, 8);
  dim3 cuda_grid((model.grid_local.i_total + 7) / 8,
                 (model.grid_local.j_total + 7) / 8,
                 (model.grid_local.k_total + 7) / 8);
  initialize_density<<<cuda_grid, cuda_block>>>(
      model.mhd_d, model.grid_local.i_total, model.grid_local.j_total,
      model.grid_local.k_total);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaDeviceSynchronize();
  model.mhd_d.qq.copy_to_host(model.mhd.qq);

  for (int i = 0; i < model.grid_local.i_total; ++i) {
    for (int j = 0; j < model.grid_local.j_total; ++j) {
      for (int k = 0; k < model.grid_local.k_total; ++k) {
        std::cout << model.mhd.qq.ro(i, j, k) << " " << i << " " << j << " " << k
                  << std::endl;
      }
    }
  }

  model.mhd_d.qq.free();

  return 0;
}
