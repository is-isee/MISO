#pragma once

#include <miso/boundary_condition_core.hpp>
#include <miso/cuda_util.cuh>
#include <miso/grid_gpu.cuh>
#include <miso/grid_view.hpp>

namespace miso {
namespace bnd {

template <typename Real>
__global__ void symmetric_kernel(Real *arr, GridView<Real> grid, Real sign,
                                 Direction direction, Side side) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  int i0_, i1_, j0_, j1_, k0_, k1_;
  range_set<Real>(i0_, i1_, j0_, j1_, k0_, k1_, direction, grid);

  if (i >= i0_ && i < i1_ && j >= j0_ && j < j1_ && k >= k0_ && k < k1_) {
    int i_ghst = i, i_trgt = i;
    int j_ghst = j, j_trgt = j;
    int k_ghst = k, k_trgt = k;

    switch (direction) {
    case Direction::X:
      symmetric_index<Real>(i, grid.i_total, grid.i_margin, i_ghst, i_trgt, side);
      break;
    case Direction::Y:
      symmetric_index<Real>(j, grid.j_total, grid.j_margin, j_ghst, j_trgt, side);
      break;
    case Direction::Z:
      symmetric_index<Real>(k, grid.k_total, grid.k_margin, k_ghst, k_trgt, side);
      break;
    }
    arr[grid.idx(i_ghst, j_ghst, k_ghst)] =
        sign * arr[grid.idx(i_trgt, j_trgt, k_trgt)];
  }
}

template <typename Real>
void symmetric(Real *arr, const GridDevice<Real> &grid, Real *fac, Real sign,
               Direction direction, Side side) {
  dim3 block_dim(8, 8, 8);
  dim3 grid_dim((grid.i_total + block_dim.x - 1) / block_dim.x,
                (grid.j_total + block_dim.y - 1) / block_dim.y,
                (grid.k_total + block_dim.z - 1) / block_dim.z);

  symmetric_kernel<Real>
      <<<grid_dim, block_dim>>>(arr, grid.view(), sign, direction, side);
  MISO_CUDA_CHECK(cudaGetLastError());
  MISO_CUDA_CHECK(cudaDeviceSynchronize());
};

}  // namespace bnd
}  // namespace miso
