#pragma once

#include <miso/cuda_util.cuh>
#include <miso/grid_view.hpp>

namespace miso {

template <typename Real> struct GridDevice {
  int i_total, j_total, k_total;
  int is, js, ks;
  int i_margin, j_margin, k_margin;
  Real min_dxyz;
  Real *x = nullptr, *y = nullptr, *z = nullptr;
  Real *dx = nullptr, *dy = nullptr, *dz = nullptr;
  Real *dxi = nullptr, *dyi = nullptr, *dzi = nullptr;

  explicit GridDevice(const Grid<Real> &grid)
      : i_total(grid.i_total), j_total(grid.j_total), k_total(grid.k_total),
        is(grid.is), js(grid.js), ks(grid.ks), i_margin(grid.i_margin),
        j_margin(grid.j_margin), k_margin(grid.k_margin),
        min_dxyz(grid.min_dxyz) {
    MISO_CUDA_CHECK(cudaMalloc(&x, sizeof(Real) * i_total));
    MISO_CUDA_CHECK(cudaMalloc(&y, sizeof(Real) * j_total));
    MISO_CUDA_CHECK(cudaMalloc(&z, sizeof(Real) * k_total));
    MISO_CUDA_CHECK(cudaMalloc(&dx, sizeof(Real) * i_total));
    MISO_CUDA_CHECK(cudaMalloc(&dy, sizeof(Real) * j_total));
    MISO_CUDA_CHECK(cudaMalloc(&dz, sizeof(Real) * k_total));
    MISO_CUDA_CHECK(cudaMalloc(&dxi, sizeof(Real) * i_total));
    MISO_CUDA_CHECK(cudaMalloc(&dyi, sizeof(Real) * j_total));
    MISO_CUDA_CHECK(cudaMalloc(&dzi, sizeof(Real) * k_total));
    copy_from_host(grid);
  }

  ~GridDevice() {
    const auto F = [](Real *&p) {
      if (p)
        MISO_CUDA_CHECK(cudaFree(p));
      p = nullptr;
    };
    // clang-format off
    F(x); F(dx); F(dxi);
    F(y); F(dy); F(dyi);
    F(z); F(dz); F(dzi);
    // clang-format on
  }

  // Shallow-const / shallow-copy
  GridView<Real> view() const noexcept { return GridView<Real>(*this); }

  void copy_from_host(const Grid<Real> &grid_h) {
    MISO_CUDA_CHECK(cudaMemcpy(x, grid_h.x.data(), sizeof(Real) * i_total,
                               cudaMemcpyHostToDevice));
    MISO_CUDA_CHECK(cudaMemcpy(y, grid_h.y.data(), sizeof(Real) * j_total,
                               cudaMemcpyHostToDevice));
    MISO_CUDA_CHECK(cudaMemcpy(z, grid_h.z.data(), sizeof(Real) * k_total,
                               cudaMemcpyHostToDevice));
    MISO_CUDA_CHECK(cudaMemcpy(dx, grid_h.dx.data(), sizeof(Real) * i_total,
                               cudaMemcpyHostToDevice));
    MISO_CUDA_CHECK(cudaMemcpy(dy, grid_h.dy.data(), sizeof(Real) * j_total,
                               cudaMemcpyHostToDevice));
    MISO_CUDA_CHECK(cudaMemcpy(dz, grid_h.dz.data(), sizeof(Real) * k_total,
                               cudaMemcpyHostToDevice));
    MISO_CUDA_CHECK(cudaMemcpy(dxi, grid_h.dxi.data(), sizeof(Real) * i_total,
                               cudaMemcpyHostToDevice));
    MISO_CUDA_CHECK(cudaMemcpy(dyi, grid_h.dyi.data(), sizeof(Real) * j_total,
                               cudaMemcpyHostToDevice));
    MISO_CUDA_CHECK(cudaMemcpy(dzi, grid_h.dzi.data(), sizeof(Real) * k_total,
                               cudaMemcpyHostToDevice));
  }

  void copy_to_host(Grid<Real> &grid_h) {
    MISO_CUDA_CHECK(cudaMemcpy(grid_h.x.data(), x, sizeof(Real) * i_total,
                               cudaMemcpyDeviceToHost));
    MISO_CUDA_CHECK(cudaMemcpy(grid_h.y.data(), y, sizeof(Real) * j_total,
                               cudaMemcpyDeviceToHost));
    MISO_CUDA_CHECK(cudaMemcpy(grid_h.z.data(), z, sizeof(Real) * k_total,
                               cudaMemcpyDeviceToHost));
    MISO_CUDA_CHECK(cudaMemcpy(grid_h.dx.data(), dx, sizeof(Real) * i_total,
                               cudaMemcpyDeviceToHost));
    MISO_CUDA_CHECK(cudaMemcpy(grid_h.dy.data(), dy, sizeof(Real) * j_total,
                               cudaMemcpyDeviceToHost));
    MISO_CUDA_CHECK(cudaMemcpy(grid_h.dz.data(), dz, sizeof(Real) * k_total,
                               cudaMemcpyDeviceToHost));
    MISO_CUDA_CHECK(cudaMemcpy(grid_h.dxi.data(), dxi, sizeof(Real) * i_total,
                               cudaMemcpyDeviceToHost));
    MISO_CUDA_CHECK(cudaMemcpy(grid_h.dyi.data(), dyi, sizeof(Real) * j_total,
                               cudaMemcpyDeviceToHost));
    MISO_CUDA_CHECK(cudaMemcpy(grid_h.dzi.data(), dzi, sizeof(Real) * k_total,
                               cudaMemcpyDeviceToHost));
  }

  // Prohibit copy and move
  GridDevice(const GridDevice &) = delete;
  GridDevice &operator=(const GridDevice &) = delete;
  GridDevice(GridDevice &&) = delete;
  GridDevice &operator=(GridDevice &&) = delete;
};

}  // namespace miso
