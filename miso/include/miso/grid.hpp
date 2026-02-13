#pragma once

#include <cassert>
#include <vector>

#include <miso/backend.hpp>
#include <miso/cuda_compat.hpp>
#include <miso/env.hpp>
#include <miso/mpi_util.hpp>

#include <miso/config.hpp>

#ifdef __CUDACC__
#include <miso/cuda_util.cuh>
#endif  // __CUDACC__

namespace miso {

/// @brief Lightweight non-owning view of Grid data.
template <typename Real> struct GridView {
  /// @brief grid number in x/y/z direction with margin
  int i_total, j_total, k_total;

  /// @brief `1` if `i/j/k_size > 1`, otherwise `0`
  int is, js, ks;

  /// @brief margin size in x/y/z direction
  int i_margin, j_margin, k_margin;

  /// @brief coordinate in x/y/z direction
  Real *x = nullptr, *y = nullptr, *z = nullptr;

  /// @brief grid spacing in x/y/z direction
  Real *dx = nullptr, *dy = nullptr, *dz = nullptr;

  /// @brief inverse grid spacing in x/y/z direction
  Real *dxi = nullptr, *dyi = nullptr, *dzi = nullptr;

  /// @brief global minimum value of dx, dy, dz
  Real min_dxyz;

  template <typename GridType>
  explicit GridView(GridType &grid) noexcept
      : i_total(grid.i_total), j_total(grid.j_total), k_total(grid.k_total),
        is(grid.is), js(grid.js), ks(grid.ks), i_margin(grid.i_margin),
        j_margin(grid.j_margin), k_margin(grid.k_margin), x(grid.x), y(grid.y),
        z(grid.z), dx(grid.dx), dy(grid.dy), dz(grid.dz), dxi(grid.dxi),
        dyi(grid.dyi), dzi(grid.dzi), min_dxyz(grid.min_dxyz) {}

  GridView(int i_total, int j_total, int k_total, int is, int js, int ks,
           int i_margin, int j_margin, int k_margin, Real min_dxyz, Real *x,
           Real *y, Real *z, Real *dx, Real *dy, Real *dz, Real *dxi, Real *dyi,
           Real *dzi) noexcept
      : i_total(i_total), j_total(j_total), k_total(k_total), is(is), js(js),
        ks(ks), i_margin(i_margin), j_margin(j_margin), k_margin(k_margin), x(x),
        y(y), z(z), dx(dx), dy(dy), dz(dz), dxi(dxi), dyi(dyi), dzi(dzi),
        min_dxyz(min_dxyz) {}
};

/// @brief Simulation grid in general backend.
template <typename Real, typename Backend = backend::Host> struct Grid;

/// @brief Simulation grid in host backend.
template <typename Real> struct Grid<Real, backend::Host> {
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
  Real x_min;
  /// @brief maximum value in x direction
  Real x_max;
  /// @brief minimum value in y direction
  Real y_min;
  /// @brief maximum value in y direction
  Real y_max;
  /// @brief minimum value in z direction
  Real z_min;
  /// @brief maximum value in z direction
  Real z_max;

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

  void global_initialize(int margin) {
    assert(i_size > 0);
    assert(j_size > 0);
    assert(k_size > 0);
    assert(margin >= 0);
    assert(x_max > x_min);
    assert(y_max > y_min);
    assert(z_max > z_min);

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
                             int i_margin, Real x_min, Real x_max) {
      x.resize(i_total);
      dx.resize(i_total);
      dxi.resize(i_total);

      Real dx0 = (x_max - x_min) / i_size;
      for (int i = 0; i < i_total; ++i) {
        // dx at i + 1/2
        dx[i] = dx0;
        dxi[i] = 1.0 / dx[i];
        if (i_size == 1) {
          dx[i] = 1.e30;
          dxi[i] = 0.0;
        }
      }

      x[i_margin] = x_min + 0.5 * dx[i_margin];
      for (int i = i_margin + 1; i < i_total; ++i) {
        x[i] = x[i - 1] + dx[i - 1];
      }
      for (int i = i_margin - 1; i >= 0; --i) {
        x[i] = x[i + 1] - dx[i];
      }
    };
    set_coordinate(x, dx, dxi, i_size, i_total, i_margin, x_min, x_max);
    set_coordinate(y, dy, dyi, j_size, j_total, j_margin, y_min, y_max);
    set_coordinate(z, dz, dzi, k_size, k_total, k_margin, z_min, z_max);

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
  Grid(int i_size_, int j_size_, int k_size_, int margin_, Real x_min_,
       Real x_max_, Real y_min_, Real y_max_, Real z_min_, Real z_max_)
      : i_size(i_size_), j_size(j_size_), k_size(k_size_), x_min(x_min_),
        x_max(x_max_), y_min(y_min_), y_max(y_max_), z_min(z_min_),
        z_max(z_max_) {
    global_initialize(margin_);
  }

  /// @brief Constructor to initialize the grid for MPI-GLOBAL geometry
  Grid(const Config &config)
      : i_size(config["grid"]["i_size"].as<int>()),
        j_size(config["grid"]["j_size"].as<int>()),
        k_size(config["grid"]["k_size"].as<int>()),
        x_min(config["grid"]["x_min"].as<Real>()),
        x_max(config["grid"]["x_max"].as<Real>()),
        y_min(config["grid"]["y_min"].as<Real>()),
        y_max(config["grid"]["y_max"].as<Real>()),
        z_min(config["grid"]["z_min"].as<Real>()),
        z_max(config["grid"]["z_max"].as<Real>()) {
    global_initialize(config["grid"]["margin"].as<int>());
  }

  ///@brief Constructor to initialize the grid for MPI-LOCAL geometry
  /// @param grid_global Global grid object
  Grid(const Grid<Real, backend::Host> &grid_global,
       const mpi::Shape &mpi_shape) {
    i_size = grid_global.i_size / mpi_shape.x_procs;
    j_size = grid_global.j_size / mpi_shape.y_procs;
    k_size = grid_global.k_size / mpi_shape.z_procs;

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

  /// @brief Copy constructor (deep copy) from another host grid.
  explicit Grid(const Grid<Real, backend::Host> &grid_h) = default;

#ifdef __CUDACC__
  /// @brief Copy constructor (deep copy) from a CUDA grid.
  explicit Grid(const Grid<Real, backend::CUDA> &grid_d)
      : i_size(grid_d.i_size), j_size(grid_d.j_size), k_size(grid_d.k_size),
        i_total(grid_d.i_total), j_total(grid_d.j_total), k_total(grid_d.k_total),
        is(grid_d.is), js(grid_d.js), ks(grid_d.ks), i_margin(grid_d.i_margin),
        j_margin(grid_d.j_margin), k_margin(grid_d.k_margin), i_stt(grid_d.i_stt),
        j_stt(grid_d.j_stt), k_stt(grid_d.k_stt), x_min(grid_d.x_min),
        y_min(grid_d.y_min), z_min(grid_d.z_min), x_max(grid_d.x_max),
        y_max(grid_d.y_max), z_max(grid_d.z_max), x(grid_d.i_total),
        y(grid_d.j_total), z(grid_d.k_total), dx(grid_d.i_total),
        dy(grid_d.j_total), dz(grid_d.k_total), dxi(grid_d.i_total),
        dyi(grid_d.j_total), dzi(grid_d.k_total), min_dxyz(grid_d.min_dxyz) {
    copy_from(grid_d);
  }
#endif  // __CUDACC__

  // Shallow-copy
  GridView<Real> view() noexcept {
    return GridView<Real>(i_total, j_total, k_total, is, js, ks, i_margin,
                          j_margin, k_margin, min_dxyz, x.data(), y.data(),
                          z.data(), dx.data(), dy.data(), dz.data(), dxi.data(),
                          dyi.data(), dzi.data());
  }
  GridView<const Real> view() const noexcept {
    return GridView<const Real>(i_total, j_total, k_total, is, js, ks, i_margin,
                                j_margin, k_margin, min_dxyz, x.data(), y.data(),
                                z.data(), dx.data(), dy.data(), dz.data(),
                                dxi.data(), dyi.data(), dzi.data());
  }
  GridView<const Real> const_view() const noexcept { return view(); }

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

#ifdef __CUDACC__
  void copy_from(const Grid<Real, backend::CUDA> &grid_d) {
    MISO_CUDA_CHECK(cudaMemcpy(x.data(), grid_d.x, sizeof(Real) * i_total,
                               cudaMemcpyDeviceToHost));
    MISO_CUDA_CHECK(cudaMemcpy(y.data(), grid_d.y, sizeof(Real) * j_total,
                               cudaMemcpyDeviceToHost));
    MISO_CUDA_CHECK(cudaMemcpy(z.data(), grid_d.z, sizeof(Real) * k_total,
                               cudaMemcpyDeviceToHost));
    MISO_CUDA_CHECK(cudaMemcpy(dx.data(), grid_d.dx, sizeof(Real) * i_total,
                               cudaMemcpyDeviceToHost));
    MISO_CUDA_CHECK(cudaMemcpy(dy.data(), grid_d.dy, sizeof(Real) * j_total,
                               cudaMemcpyDeviceToHost));
    MISO_CUDA_CHECK(cudaMemcpy(dz.data(), grid_d.dz, sizeof(Real) * k_total,
                               cudaMemcpyDeviceToHost));
    MISO_CUDA_CHECK(cudaMemcpy(dxi.data(), grid_d.dxi, sizeof(Real) * i_total,
                               cudaMemcpyDeviceToHost));
    MISO_CUDA_CHECK(cudaMemcpy(dyi.data(), grid_d.dyi, sizeof(Real) * j_total,
                               cudaMemcpyDeviceToHost));
    MISO_CUDA_CHECK(cudaMemcpy(dzi.data(), grid_d.dzi, sizeof(Real) * k_total,
                               cudaMemcpyDeviceToHost));
  }
#endif  // __CUDACC__
};

#ifdef __CUDACC__
/// @brief Simulation grid in CUDA backend.
template <typename Real> struct Grid<Real, backend::CUDA> {
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
  Real x_min;
  /// @brief maximum value in x direction
  Real x_max;
  /// @brief minimum value in y direction
  Real y_min;
  /// @brief maximum value in y direction
  Real y_max;
  /// @brief minimum value in z direction
  Real z_min;
  /// @brief maximum value in z direction
  Real z_max;

  /// @brief coordinate in x/y/z direction
  Real *x = nullptr, *y = nullptr, *z = nullptr;
  /// @brief grid spacing in x/y/z direction
  Real *dx = nullptr, *dy = nullptr, *dz = nullptr;
  /// @brief inverse grid spacing in x/y/z direction
  Real *dxi = nullptr, *dyi = nullptr, *dzi = nullptr;

  /// @brief global minimum value of dx, dy, dz
  Real min_dxyz;

  explicit Grid(const Grid<Real, backend::Host> &grid)
      : i_size(grid.i_size), j_size(grid.j_size), k_size(grid.k_size),
        i_total(grid.i_total), j_total(grid.j_total), k_total(grid.k_total),
        is(grid.is), js(grid.js), ks(grid.ks), i_margin(grid.i_margin),
        j_margin(grid.j_margin), k_margin(grid.k_margin), i_stt(grid.i_stt),
        j_stt(grid.j_stt), k_stt(grid.k_stt), x_min(grid.x_min),
        x_max(grid.x_max), y_min(grid.y_min), y_max(grid.y_max),
        z_min(grid.z_min), z_max(grid.z_max), min_dxyz(grid.min_dxyz) {
    MISO_CUDA_CHECK(cudaMalloc(&x, sizeof(Real) * i_total));
    MISO_CUDA_CHECK(cudaMalloc(&y, sizeof(Real) * j_total));
    MISO_CUDA_CHECK(cudaMalloc(&z, sizeof(Real) * k_total));
    MISO_CUDA_CHECK(cudaMalloc(&dx, sizeof(Real) * i_total));
    MISO_CUDA_CHECK(cudaMalloc(&dy, sizeof(Real) * j_total));
    MISO_CUDA_CHECK(cudaMalloc(&dz, sizeof(Real) * k_total));
    MISO_CUDA_CHECK(cudaMalloc(&dxi, sizeof(Real) * i_total));
    MISO_CUDA_CHECK(cudaMalloc(&dyi, sizeof(Real) * j_total));
    MISO_CUDA_CHECK(cudaMalloc(&dzi, sizeof(Real) * k_total));
    copy_from(grid);
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

  // Shallow-copy
  GridView<Real> view() noexcept { return GridView<Real>(*this); }
  GridView<const Real> view() const noexcept {
    return GridView<const Real>(*this);
  }
  GridView<const Real> const_view() const noexcept { return view(); }

  void copy_from(const Grid<Real, backend::Host> &grid_h) {
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

  // Prohibit copy and move
  Grid<Real, backend::CUDA>(const Grid<Real, backend::CUDA> &) = delete;
  Grid<Real, backend::CUDA> &
  operator=(const Grid<Real, backend::CUDA> &) = delete;
  Grid<Real, backend::CUDA>(Grid<Real, backend::CUDA> &&) = delete;
  Grid<Real, backend::CUDA> &operator=(Grid<Real, backend::CUDA> &&) = delete;
};
#endif  // __CUDACC__

}  // namespace miso
