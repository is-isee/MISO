#pragma once
#include <cassert>
#include <vector>

#include <miso/array3d_cpu.hpp>
#include <miso/config.hpp>
#include <miso/mpi_manager.hpp>

namespace miso {

/// @brief  Grid class for CPU version
/// @tparam Real
template <typename Real> struct Grid {
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

  /// @brief  minimum value in x direction
  Real xmin;
  /// @brief  maximum value in x direction
  Real xmax;
  /// @brief minimum value in y direction
  Real ymin;
  /// @brief maximum value in y direction
  Real ymax;
  /// @brief minimum value in z direction
  Real zmin;
  /// @brief maximum value in z direction
  Real zmax;

  /// @brief  coordinate in x direction
  std::vector<Real> x;
  /// @brief  coordinate in y direction
  std::vector<Real> y;
  /// @brief  coordinate in z direction
  std::vector<Real> z;

  /// @brief  grid spacing in x direction
  std::vector<Real> dx;
  /// @brief  grid spacing in y direction
  std::vector<Real> dy;
  /// @brief  grid spacing in z direction
  std::vector<Real> dz;

  /// @brief  inverse grid spacing in x direction
  std::vector<Real> dxi;
  /// @brief  inverse grid spacing in y direction
  std::vector<Real> dyi;
  /// @brief  inverse grid spacing in z direction
  std::vector<Real> dzi;

  /// @brief global minimum value of dx, dy, dz
  Real min_dxyz;

  /// @brief mask array
  Array3D<Real> mask;

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
  /// @param yaml_obj
  Grid(const YAML::Node &yaml_obj)
      : i_size(yaml_obj["grid"]["i_size"].template as<int>()),
        j_size(yaml_obj["grid"]["j_size"].template as<int>()),
        k_size(yaml_obj["grid"]["k_size"].template as<int>()),
        margin(yaml_obj["grid"]["margin"].template as<int>()),
        xmin(yaml_obj["grid"]["xmin"].template as<Real>()),
        xmax(yaml_obj["grid"]["xmax"].template as<Real>()),
        ymin(yaml_obj["grid"]["ymin"].template as<Real>()),
        ymax(yaml_obj["grid"]["ymax"].template as<Real>()),
        zmin(yaml_obj["grid"]["zmin"].template as<Real>()),
        zmax(yaml_obj["grid"]["zmax"].template as<Real>()) {
    global_initialize();
  }

  ///@brief Constructor to initialize the grid for MPI-LOCAL geometry
  /// @param grid_global Global grid object
  Grid(const Grid<Real> &grid_global, const MPITopology &mpi) {
    i_size = grid_global.i_size / mpi.x_procs;
    j_size = grid_global.j_size / mpi.y_procs;
    k_size = grid_global.k_size / mpi.z_procs;

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

    i_stt = mpi.coord[0] * i_size;
    j_stt = mpi.coord[1] * j_size;
    k_stt = mpi.coord[2] * k_size;

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

    mask.allocate(i_total, j_total, k_total);

    // set mask value, default is 1 (fluid)
    for (int i = 0; i < i_total; ++i) {
      for (int j = 0; j < j_total; ++j) {
        for (int k = 0; k < k_total; ++k) {
          mask(i, j, k) = 1;
        }
      }
    }
  }

  /// @brief save grid data to a binary file
  /// @param config
  void save(const Config &config) const {
    if (config.mpi_rt.is_root()) {
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
