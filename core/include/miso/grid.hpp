#pragma once

#include <cassert>
#include <vector>

#include <miso/cuda_compat.hpp>
#include <miso/env.hpp>
#include <miso/mpi_util.hpp>
#include <miso/policy.hpp>

#include <miso/config.hpp>

#ifdef USE_CUDA
#include <miso/cuda_util.cuh>
#endif  // USE_CUDA

namespace miso {

/// @brief Lightweight non-owning view of Grid data.
template <typename Real> struct GridView {
  int i_total, j_total, k_total;
  int is, js, ks;
  int i_margin, j_margin, k_margin;
  Real min_dxyz;

  Real *x = nullptr, *y = nullptr, *z = nullptr;
  Real *dx = nullptr, *dy = nullptr, *dz = nullptr;
  Real *dxi = nullptr, *dyi = nullptr, *dzi = nullptr;

  template <typename GridType>
  explicit GridView(GridType &grid) noexcept
      : i_total(grid.i_total), j_total(grid.j_total), k_total(grid.k_total),
        is(grid.is), js(grid.js), ks(grid.ks), i_margin(grid.i_margin),
        j_margin(grid.j_margin), k_margin(grid.k_margin), min_dxyz(grid.min_dxyz),
        x(grid.x), y(grid.y), z(grid.z), dx(grid.dx), dy(grid.dy), dz(grid.dz),
        dxi(grid.dxi), dyi(grid.dyi), dzi(grid.dzi) {}

  __device__ inline int idx(int i, int j, int k) const {
    return (i * j_total + j) * k_total + k;
  }
};

/// @brief Simulation grid in general memory space.
template <typename Real, typename MemorySpace = HostSpace> struct Grid;

/// @brief Simulation grid in host memory.
template <typename Real> struct Grid<Real, HostSpace> {
  /// @brief  grid number in x direction without margin
  int i_size;
  /// @brief  grid number in y direction without margin
  int j_size;
  /// @brief  grid number in z direction without margin
  int k_size;

  /// @brief  grid number in x direction with margin
  int i_total;
  /// @brief  grid number in y direction with margin
  int j_total;
  /// @brief  grid number in z direction with margin
  int k_total;

  /// @brief `1` if `i_size > 1`, otherwise `0`
  int is;
  /// @brief `1` if `j_size > 1`, otherwise `0`
  int js;
  /// @brief `1` if `k_size > 1`, otherwise `0`
  int ks;

  /// @brief  margin size for numerical scheme
  int margin;
  /// @brief  margin size in x direction
  int i_margin;
  /// @brief  margin size in y direction
  int j_margin;
  /// @brief  margin size in z direction
  int k_margin;

  /// @brief starting index in x direction (for MPI calculation)
  int i_stt;
  /// @brief starting index in y direction (for MPI calculation)
  int j_stt;
  /// @brief starting index in z direction (for MPI calculation)
  int k_stt;

  /// @brief minimum value in x direction
  Real xmin;
  /// @brief maximum value in x direction
  Real xmax;
  /// @brief minimum value in y direction
  Real ymin;
  /// @brief maximum value in y direction
  Real ymax;
  /// @brief minimum value in z direction
  Real zmin;
  /// @brief maximum value in z direction
  Real zmax;

  /// @brief coordinate in x direction
  std::vector<Real> x;
  /// @brief coordinate in y direction
  std::vector<Real> y;
  /// @brief coordinate in z direction
  std::vector<Real> z;

  /// @brief grid spacing in x direction
  std::vector<Real> dx;
  /// @brief grid spacing in y direction
  std::vector<Real> dy;
  /// @brief grid spacing in z direction
  std::vector<Real> dz;

  /// @brief inverse grid spacing in x direction
  std::vector<Real> dxi;
  /// @brief inverse grid spacing in y direction
  std::vector<Real> dyi;
  /// @brief inverse grid spacing in z direction
  std::vector<Real> dzi;

  /// @brief global minimum value of dx, dy, dz
  Real min_dxyz;

  inline int idx(int i, int j, int k) const {
    return (i * j_total + j) * k_total + k;
  }

  void global_initialize() {
    assert(i_size > 0);
    assert(j_size > 0);
    assert(k_size > 0);
    assert(margin >= 0);
    assert(xmax > xmin);
    assert(ymax > ymin);
    assert(zmax > zmin);

    is = i_size > 1 ? 1 : 0;
    js = j_size > 1 ? 1 : 0;
    ks = k_size > 1 ? 1 : 0;

    i_margin = margin * is;
    j_margin = margin * js;
    k_margin = margin * ks;

    i_total = i_size + 2 * i_margin;
    j_total = j_size + 2 * j_margin;
    k_total = k_size + 2 * k_margin;

    i_stt = 0;
    j_stt = 0;
    k_stt = 0;

    auto set_coordinate = [](std::vector<Real> &x, std::vector<Real> &dx,
                             std::vector<Real> &dxi, int i_size, int i_total,
                             int i_margin, Real xmin, Real xmax) {
      x.resize(i_total);
      dx.resize(i_total);
      dxi.resize(i_total);

      Real dx0 = (xmax - xmin) / i_size;
      for (int i = 0; i < i_total; ++i) {
        // dx at i + 1/2
        dx[i] = dx0;
        dxi[i] = 1.0 / dx[i];
        if (i_size == 1) {
          dx[i] = 1.e30;
          dxi[i] = 0.0;
        }
      }

      x[i_margin] = xmin + 0.5 * dx[i_margin];
      for (int i = i_margin + 1; i < i_total; ++i) {
        x[i] = x[i - 1] + dx[i - 1];
      }
      for (int i = i_margin - 1; i >= 0; --i) {
        x[i] = x[i + 1] - dx[i];
      }
    };
    set_coordinate(x, dx, dxi, i_size, i_total, i_margin, xmin, xmax);
    set_coordinate(y, dy, dyi, j_size, j_total, j_margin, ymin, ymax);
    set_coordinate(z, dz, dzi, k_size, k_total, k_margin, zmin, zmax);

    Real min_dx = 1.e30, min_dy = 1.e30, min_dz = 1.e30;
    /// TODO: additional communication is required for MPI version
    for (int i = 0; i < i_total; ++i) {
      min_dx = std::min<Real>(min_dx, dx[i]);
    }
    for (int j = 0; j < j_total; ++j) {
      min_dy = std::min<Real>(min_dy, dy[j]);
    }
    for (int k = 0; k < k_total; ++k) {
      min_dz = std::min<Real>(min_dz, dz[k]);
    }
    min_dxyz = std::min<Real>({min_dx, min_dy, min_dz});
  }

  // global settings
  Grid(int i_size_, int j_size_, int k_size_, int margin_, Real xmin_, Real xmax_,
       Real ymin_, Real ymax_, Real zmin_, Real zmax_)
      : i_size(i_size_), j_size(j_size_), k_size(k_size_), margin(margin_),
        xmin(xmin_), xmax(xmax_), ymin(ymin_), ymax(ymax_), zmin(zmin_),
        zmax(zmax_) {
    global_initialize();
  }

  /// @brief Constructor to initialize the grid for MPI-GLOBAL geometry
  Grid(const Config &config)
      : i_size(config["grid"]["i_size"].template as<int>()),
        j_size(config["grid"]["j_size"].template as<int>()),
        k_size(config["grid"]["k_size"].template as<int>()),
        margin(config["grid"]["margin"].template as<int>()),
        xmin(config["grid"]["xmin"].template as<Real>()),
        xmax(config["grid"]["xmax"].template as<Real>()),
        ymin(config["grid"]["ymin"].template as<Real>()),
        ymax(config["grid"]["ymax"].template as<Real>()),
        zmin(config["grid"]["zmin"].template as<Real>()),
        zmax(config["grid"]["zmax"].template as<Real>()) {
    global_initialize();
  }

  ///@brief Constructor to initialize the grid for MPI-LOCAL geometry
  /// @param grid_global Global grid object
  Grid(const Grid<Real, HostSpace> &grid_global, const mpi::Shape &mpi_shape) {
    i_size = grid_global.i_size / mpi_shape.x_procs;
    j_size = grid_global.j_size / mpi_shape.y_procs;
    k_size = grid_global.k_size / mpi_shape.z_procs;

    margin = grid_global.margin;

    is = grid_global.is;
    js = grid_global.js;
    ks = grid_global.ks;

    i_margin = grid_global.i_margin;
    j_margin = grid_global.j_margin;
    k_margin = grid_global.k_margin;

    i_total = i_size + 2 * i_margin;
    j_total = j_size + 2 * j_margin;
    k_total = k_size + 2 * k_margin;

    i_stt = mpi_shape.coord[0] * i_size;
    j_stt = mpi_shape.coord[1] * j_size;
    k_stt = mpi_shape.coord[2] * k_size;

    x.resize(i_total);
    dx.resize(i_total);
    dxi.resize(i_total);
    y.resize(j_total);
    dy.resize(j_total);
    dyi.resize(j_total);
    z.resize(k_total);
    dz.resize(k_total);
    dzi.resize(k_total);

    for (int i = 0; i < i_total; ++i) {
      x[i] = grid_global.x[i_stt + i];
      dx[i] = grid_global.dx[i_stt + i];
      dxi[i] = grid_global.dxi[i_stt + i];
    }

    for (int j = 0; j < j_total; ++j) {
      y[j] = grid_global.y[j_stt + j];
      dy[j] = grid_global.dy[j_stt + j];
      dyi[j] = grid_global.dyi[j_stt + j];
    }

    for (int k = 0; k < k_total; ++k) {
      z[k] = grid_global.z[k_stt + k];
      dz[k] = grid_global.dz[k_stt + k];
      dzi[k] = grid_global.dzi[k_stt + k];
    }
    min_dxyz = grid_global.min_dxyz;
  }

  /// @brief save grid data to a binary file
  /// @param config
  void save(const Config &config) const {
    if (mpi::is_root()) {
      std::ofstream ofs_bin(config.save_dir + "/grid.bin", std::ios::binary);
      assert(ofs_bin.is_open());

      auto write_array = [&ofs_bin](const std::vector<Real> &x) {
        ofs_bin.write(reinterpret_cast<const char *>(x.data()),
                      sizeof(Real) * x.size());
      };
      write_array(x);
      write_array(y);
      write_array(z);
    }
  }
};

#ifdef USE_CUDA
/// @brief Simulation grid in CUDA device memory.
template <typename Real> struct Grid<Real, CUDASpace> {
  int i_total, j_total, k_total;
  int is, js, ks;
  int i_margin, j_margin, k_margin;
  Real min_dxyz;
  Real *x = nullptr, *y = nullptr, *z = nullptr;
  Real *dx = nullptr, *dy = nullptr, *dz = nullptr;
  Real *dxi = nullptr, *dyi = nullptr, *dzi = nullptr;

  explicit Grid(const Grid<Real, HostSpace> &grid)
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

  ~Grid() {
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

  void copy_from_host(const Grid<Real, HostSpace> &grid_h) {
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

  void copy_to_host(Grid<Real, HostSpace> &grid_h) {
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
  Grid<Real, CUDASpace>(const Grid<Real, CUDASpace> &) = delete;
  Grid<Real, CUDASpace> &operator=(const Grid<Real, CUDASpace> &) = delete;
  Grid<Real, CUDASpace>(Grid<Real, CUDASpace> &&) = delete;
  Grid<Real, CUDASpace> &operator=(Grid<Real, CUDASpace> &&) = delete;
};
#endif  // USE_CUDA

}  // namespace miso
