#pragma once

#include <miso/cuda_compat.hpp>
#include <miso/env.hpp>
#include <miso/mpi_util.hpp>

#include <miso/config.hpp>

#ifdef __CUDACC__
#include <miso/cuda_util.cuh>
#endif  // __CUDACC__

namespace miso {

/// @brief Generator of local Grid objects with MPI topology
template <typename Real> struct GridGenerator {
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

  inline int idx(int i, int j, int k) const {
    return (i * j_total + j) * k_total + k;
  }

  void global_initialize() {
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
      : i_size(i_size_), j_size(j_size_), k_size(k_size_), margin(margin_),
        x_min(x_min_), x_max(x_max_), y_min(y_min_), y_max(y_max_), z_min(z_min_),
        z_max(z_max_) {
    global_initialize();
  }

  /// @brief Constructor to initialize the grid for MPI-GLOBAL geometry
  Grid(const Config &config)
      : i_size(config["grid"]["i_size"].as<int>()),
        j_size(config["grid"]["j_size"].as<int>()),
        k_size(config["grid"]["k_size"].as<int>()),
        margin(config["grid"]["margin"].as<int>()),
        x_min(config["grid"]["x_min"].as<Real>()),
        x_max(config["grid"]["x_max"].as<Real>()),
        y_min(config["grid"]["y_min"].as<Real>()),
        y_max(config["grid"]["y_max"].as<Real>()),
        z_min(config["grid"]["z_min"].as<Real>()),
        z_max(config["grid"]["z_max"].as<Real>()) {
    global_initialize();
  }

  ///@brief Constructor to initialize the grid for MPI-LOCAL geometry
  /// @param grid_global Global grid object
  Grid(const Grid<Real> &grid_global, const mpi::Shape &mpi_shape) {
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

}  // namespace miso
