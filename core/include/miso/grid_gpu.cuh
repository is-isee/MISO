#pragma once
#include <cuda_runtime.h>

#include <miso/cuda_manager.cuh>

namespace miso {

template <typename Real> __host__ __device__ struct GridDevice {
  int i_total, j_total, k_total;
  int is, js, ks;
  int i_margin, j_margin, k_margin;
  Real min_dxyz;

  Real *x = nullptr, *y = nullptr, *z = nullptr;
  Real *dx = nullptr, *dy = nullptr, *dz = nullptr;
  Real *dxi = nullptr, *dyi = nullptr, *dzi = nullptr;

  Real *mask = nullptr;

  __device__ inline int idx(int i, int j, int k) const {
    return (i * j_total + j) * k_total + k;
  }

  GridDevice(const Grid<Real> &grid)
      : i_total(grid.i_total), j_total(grid.j_total), k_total(grid.k_total),
        is(grid.is), js(grid.js), ks(grid.ks), i_margin(grid.i_margin),
        j_margin(grid.j_margin), k_margin(grid.k_margin),
        min_dxyz(grid.min_dxyz) {

    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaMalloc(&x, sizeof(Real) * i_total));
    CUDA_CHECK(cudaFree(y));
    CUDA_CHECK(cudaMalloc(&y, sizeof(Real) * j_total));
    CUDA_CHECK(cudaFree(z));
    CUDA_CHECK(cudaMalloc(&z, sizeof(Real) * k_total));
    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaMalloc(&dx, sizeof(Real) * i_total));
    CUDA_CHECK(cudaFree(dy));
    CUDA_CHECK(cudaMalloc(&dy, sizeof(Real) * j_total));
    CUDA_CHECK(cudaFree(dz));
    CUDA_CHECK(cudaMalloc(&dz, sizeof(Real) * k_total));
    CUDA_CHECK(cudaFree(dxi));
    CUDA_CHECK(cudaMalloc(&dxi, sizeof(Real) * i_total));
    CUDA_CHECK(cudaFree(dyi));
    CUDA_CHECK(cudaMalloc(&dyi, sizeof(Real) * j_total));
    CUDA_CHECK(cudaFree(dzi));
    CUDA_CHECK(cudaMalloc(&dzi, sizeof(Real) * k_total));
    CUDA_CHECK(cudaFree(mask));
    CUDA_CHECK(cudaMalloc(&mask, sizeof(Real) * i_total * j_total * k_total));
  }

  ~GridDevice() {}
  void free() {
    if (x)
      CUDA_CHECK(cudaFree(x));
    x = nullptr;
    if (y)
      CUDA_CHECK(cudaFree(y));
    y = nullptr;
    if (z)
      CUDA_CHECK(cudaFree(z));
    z = nullptr;
    if (dx)
      CUDA_CHECK(cudaFree(dx));
    dx = nullptr;
    if (dy)
      CUDA_CHECK(cudaFree(dy));
    dy = nullptr;
    if (dz)
      CUDA_CHECK(cudaFree(dz));
    dz = nullptr;
    if (dxi)
      CUDA_CHECK(cudaFree(dxi));
    dxi = nullptr;
    if (dyi)
      CUDA_CHECK(cudaFree(dyi));
    dyi = nullptr;
    if (dzi)
      CUDA_CHECK(cudaFree(dzi));
    dzi = nullptr;
  }

  GridDevice(const GridDevice &) = default;
  GridDevice &operator=(const GridDevice &) = default;

  void copy_from_host(const Grid<Real> &grid_h) {
    CUDA_CHECK(cudaMemcpy(x, grid_h.x.data(), sizeof(Real) * i_total,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(y, grid_h.y.data(), sizeof(Real) * j_total,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(z, grid_h.z.data(), sizeof(Real) * k_total,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dx, grid_h.dx.data(), sizeof(Real) * i_total,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy, grid_h.dy.data(), sizeof(Real) * j_total,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dz, grid_h.dz.data(), sizeof(Real) * k_total,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dxi, grid_h.dxi.data(), sizeof(Real) * i_total,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dyi, grid_h.dyi.data(), sizeof(Real) * j_total,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dzi, grid_h.dzi.data(), sizeof(Real) * k_total,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mask, grid_h.mask.data(),
                          sizeof(Real) * i_total * j_total * k_total,
                          cudaMemcpyHostToDevice));
  }

  void copy_to_host(Grid<Real> &grid_h) {
    CUDA_CHECK(cudaMemcpy(grid_h.x.data(), x, sizeof(Real) * i_total,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(grid_h.y.data(), y, sizeof(Real) * j_total,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(grid_h.z.data(), z, sizeof(Real) * k_total,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(grid_h.dx.data(), dx, sizeof(Real) * i_total,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(grid_h.dy.data(), dy, sizeof(Real) * j_total,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(grid_h.dz.data(), dz, sizeof(Real) * k_total,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(grid_h.dxi.data(), dxi, sizeof(Real) * i_total,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(grid_h.dyi.data(), dyi, sizeof(Real) * j_total,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(grid_h.dzi.data(), dzi, sizeof(Real) * k_total,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(grid_h.mask.data(), mask,
                          sizeof(Real) * i_total * j_total * k_total,
                          cudaMemcpyDeviceToHost));
  }
};

}  // namespace miso
