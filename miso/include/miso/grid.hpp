#pragma once

#include <cassert>
#include <exception>
#include <utility>
#include <vector>

#include "backend.hpp"
#include "cuda_compat.hpp"
#include "env.hpp"
#include "mpi_util.hpp"

#include "config.hpp"

#ifdef __CUDACC__
#include "cuda_util.cuh"
#endif  // __CUDACC__

namespace miso {

/// @brief Provider of uniform grid axis.
template <typename Real> struct UniformAxisProvider {
  Real min_value, max_value;
  int size;

  /// @brief Spacing of i-th grid point.
  Real space(int) const { return (max_value - min_value) / Real(size); }

  /// @brief Location of i-th grid point.
  Real loc(int i) const { return min_value + (Real(i) + Real(0.5)) * space(i); }
};

/// @brief 1D grid along an axis.
template <typename Real> struct AxisGrid {
  /// @brief number of grid points without margin
  int size;

  /// @brief number of grid points with margin
  int total;

  /// @brief margin size
  int margin;

  /// @brief stride indicator (1 if size > 1, else 0)
  int stride;

  /// @brief beginning index
  int begin;

  /// @brief ending index
  int end;

  /// @brief offset of indexing (from global to local)
  int offset;

  /// @brief grid point locations
  std::vector<Real> s;

  /// @brief grid spacings
  std::vector<Real> ds;

  /// @brief inverse grid spacings
  std::vector<Real> dsi;

  /// @brief Construct axis grid.
  template <typename AxisProvider>
  explicit AxisGrid(int size_, int margin_, int offset_, const AxisProvider &ap) {
    size = size_;
    stride = size > 1 ? 1 : 0;
    margin = margin_ * stride;
    total = size + 2 * margin;
    begin = margin;
    end = margin + size;
    offset = offset_;
    assert(size > 0);
    assert(margin >= 0);
    assert(begin >= 0);
    assert(end - begin == size);

    s.resize(total);
    ds.resize(total);
    dsi.resize(total);

    if (size == 1) {
      assert(margin == 0);
      assert(offset == 0);
      s[0] = ap.loc(0);
      ds[0] = 1.e30;  // very large value
      dsi[0] = 0.0;
      return;
    }

    for (int i = 0; i < total; ++i) {
      const int ig = i - begin + offset;
      s[i] = ap.loc(ig);
      ds[i] = ap.space(ig);
      dsi[i] = 1.0 / ds[i];
    }
  }
};

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

  void assemble_from_axes(AxisGrid<Real> &&x_grid, AxisGrid<Real> &&y_grid,
                          AxisGrid<Real> &&z_grid) {
    i_total = x_grid.total;
    j_total = y_grid.total;
    k_total = z_grid.total;

    is = x_grid.stride;
    js = y_grid.stride;
    ks = z_grid.stride;

    i_margin = x_grid.margin;
    j_margin = y_grid.margin;
    k_margin = z_grid.margin;

    x = std::move(x_grid.s);
    y = std::move(y_grid.s);
    z = std::move(z_grid.s);

    dx = std::move(x_grid.ds);
    dy = std::move(y_grid.ds);
    dz = std::move(z_grid.ds);

    dxi = std::move(x_grid.dsi);
    dyi = std::move(y_grid.dsi);
    dzi = std::move(z_grid.dsi);

    /// Compute global minimum value of dx, dy, dz
    /// TODO: Assuming uniform spacing.
    min_dxyz = util::min3(dx[0], dy[0], dz[0]);
  }

  void check_margin(int margin) const {
    if (margin < 1) {
      throw std::runtime_error("Invalid margin size: margin < 1");
    }
  }

  void check_size(int i_size, int j_size, int k_size) const {
    if (i_size <= 0 || j_size <= 0 || k_size <= 0) {
      throw std::runtime_error("Invalid grid size: i_size <= 0 or j_size <= 0 or "
                               "k_size <= 0");
    }
  }

  void check_range(Real x_min, Real x_max, Real y_min, Real y_max, Real z_min,
                   Real z_max) const {
    if (x_min >= x_max || y_min >= y_max || z_min >= z_max) {
      throw std::runtime_error("Invalid grid range: x_min >= x_max or y_min >= "
                               "y_max or z_min >= z_max");
    }
  }

  void check_divisibility(int i_size_g, int j_size_g, int k_size_g,
                          const mpi::Shape &mpi_shape) const {
    if (i_size_g % mpi_shape.x_procs != 0 || j_size_g % mpi_shape.y_procs != 0 ||
        k_size_g % mpi_shape.z_procs != 0) {
      printf("i_size_g = %d, j_size_g = %d, k_size_g = %d\n", i_size_g, j_size_g,
             k_size_g);
      printf("x_procs = %d, y_procs = %d, z_procs = %d\n", mpi_shape.x_procs,
             mpi_shape.y_procs, mpi_shape.z_procs);
      throw std::runtime_error(
          "Grid size is not divisible by the number of MPI processes.");
    }
  }

  /// @brief Construct simulation grid.
  explicit Grid(const Config &config, const mpi::Shape &mpi_shape)
      : x_min(config["grid"]["x_min"].as<Real>()),
        x_max(config["grid"]["x_max"].as<Real>()),
        y_min(config["grid"]["y_min"].as<Real>()),
        y_max(config["grid"]["y_max"].as<Real>()),
        z_min(config["grid"]["z_min"].as<Real>()),
        z_max(config["grid"]["z_max"].as<Real>()) {
    const int margin = config["grid"]["margin"].as<int>();
    const int i_size_g = config["grid"]["i_size"].as<int>();
    const int j_size_g = config["grid"]["j_size"].as<int>();
    const int k_size_g = config["grid"]["k_size"].as<int>();

    i_size = i_size_g / mpi_shape.x_procs;
    j_size = j_size_g / mpi_shape.y_procs;
    k_size = k_size_g / mpi_shape.z_procs;

    auto x_provider = UniformAxisProvider<Real>{x_min, x_max, i_size_g};
    auto y_provider = UniformAxisProvider<Real>{y_min, y_max, j_size_g};
    auto z_provider = UniformAxisProvider<Real>{z_min, z_max, k_size_g};

    const int i_offset = mpi_shape.coord[0] * i_size;
    const int j_offset = mpi_shape.coord[1] * j_size;
    const int k_offset = mpi_shape.coord[2] * k_size;

    auto x_grid = AxisGrid<Real>(i_size, margin, i_offset, x_provider);
    auto y_grid = AxisGrid<Real>(j_size, margin, j_offset, y_provider);
    auto z_grid = AxisGrid<Real>(k_size, margin, k_offset, z_provider);
    assemble_from_axes(std::move(x_grid), std::move(y_grid), std::move(z_grid));

    // Argument check
    check_margin(margin);
    check_divisibility(i_size_g, j_size_g, k_size_g, mpi_shape);
    check_size(i_size, j_size, k_size);
    check_range(x_min, x_max, y_min, y_max, z_min, z_max);
  }

  /// @brief Construct simulation grid from axis grids (serial version).
  explicit Grid(int i_size_, int j_size_, int k_size_, int margin_, Real x_min_,
                Real x_max_, Real y_min_, Real y_max_, Real z_min_, Real z_max_)
      : i_size(i_size_), j_size(j_size_), k_size(k_size_), x_min(x_min_),
        x_max(x_max_), y_min(y_min_), y_max(y_max_), z_min(z_min_),
        z_max(z_max_) {

    auto x_provider = UniformAxisProvider<Real>{x_min, x_max, i_size};
    auto y_provider = UniformAxisProvider<Real>{y_min, y_max, j_size};
    auto z_provider = UniformAxisProvider<Real>{z_min, z_max, k_size};

    constexpr int offset = 0;  // No offset for serial version
    auto x_grid = AxisGrid<Real>(i_size, margin_, offset, x_provider);
    auto y_grid = AxisGrid<Real>(j_size, margin_, offset, y_provider);
    auto z_grid = AxisGrid<Real>(k_size, margin_, offset, z_provider);

    assemble_from_axes(std::move(x_grid), std::move(y_grid), std::move(z_grid));

    // Argument check
    check_margin(margin_);
    check_size(i_size_, j_size_, k_size_);
    check_range(x_min_, x_max_, y_min_, y_max_, z_min_, z_max_);
  }

  /// @brief Copy constructor.
  explicit Grid(const Grid<Real, backend::Host> &) = default;

  /// @brief Copy assignment operator.
  Grid<Real, backend::Host> &
  operator=(const Grid<Real, backend::Host> &) = default;

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
      // Generate global grid for saving
      /// TODO: This should be implemented by MPI_gather.
      const int margin = config["grid"]["margin"].as<int>();
      const int i_size_g = config["grid"]["i_size"].as<int>();
      const int j_size_g = config["grid"]["j_size"].as<int>();
      const int k_size_g = config["grid"]["k_size"].as<int>();

      auto x_provider = UniformAxisProvider<Real>{x_min, x_max, i_size_g};
      auto y_provider = UniformAxisProvider<Real>{y_min, y_max, j_size_g};
      auto z_provider = UniformAxisProvider<Real>{z_min, z_max, k_size_g};

      auto x_grid = AxisGrid<Real>(i_size_g, margin, 0, x_provider);
      auto y_grid = AxisGrid<Real>(j_size_g, margin, 0, y_provider);
      auto z_grid = AxisGrid<Real>(k_size_g, margin, 0, z_provider);

      // Save global grid to a binary file
      std::ofstream ofs(config.save_dir + "/grid.bin", std::ios::binary);
      assert(ofs.is_open());

      auto write_array = [&ofs](const std::vector<Real> &x) {
        ofs.write(reinterpret_cast<const char *>(x.data()),
                  sizeof(Real) * x.size());
      };
      write_array(x_grid.s);
      write_array(y_grid.s);
      write_array(z_grid.s);
    }
  }

#ifdef __CUDACC__
  void copy_from(const Grid<Real, backend::CUDA> &grid_d) {
    assert(i_total == grid_d.i_total);
    assert(j_total == grid_d.j_total);
    assert(k_total == grid_d.k_total);
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
        j_margin(grid.j_margin), k_margin(grid.k_margin), x_min(grid.x_min),
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

  /// @brief Return a lightweight view.
  GridView<Real> view() noexcept { return GridView<Real>(*this); }

  /// @brief Return a constant lightweight view.
  GridView<const Real> view() const noexcept {
    return GridView<const Real>(*this);
  }

  /// @brief Return a constant lightweight view.
  GridView<const Real> const_view() const noexcept { return view(); }

  void copy_from(const Grid<Real, backend::Host> &grid_h) {
    assert(i_total == grid_h.i_total);
    assert(j_total == grid_h.j_total);
    assert(k_total == grid_h.k_total);
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

  // Prohibit copy semantics
  Grid<Real, backend::CUDA>(const Grid<Real, backend::CUDA> &) = delete;
  Grid<Real, backend::CUDA> &
  operator=(const Grid<Real, backend::CUDA> &) = delete;

  // Prohibit move semantics
  Grid<Real, backend::CUDA>(Grid<Real, backend::CUDA> &&) = delete;
  Grid<Real, backend::CUDA> &operator=(Grid<Real, backend::CUDA> &&) = delete;
};
#endif  // __CUDACC__

}  // namespace miso
