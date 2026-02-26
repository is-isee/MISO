#pragma once
#include "array1d.hpp"
#include "array2d.hpp"
#include "array3d.hpp"
#include "execution.hpp"
#include "utility.hpp"

namespace miso {

// Interpolate a value at x for 1D table.
template <class T>
__device__ __host__ T interpolate_uniform_table1d(const T *table, T x_min,
                                                  T dx_inv, T x, int n) noexcept {
  assert(n > 1);
  T fi = (x - x_min) * dx_inv;
  fi = util::clamp<T>(fi, T(0), T(n - 1));
  int i = util::clamp<int>(static_cast<int>(fi), 0, n - 2);
  T t = fi - T(i);
  return (T(1) - t) * table[i] + t * table[i + 1];
}

/// @brief Interpolator for 1D (x) table in the uniform grid.
/// @details The table is assumed to be uniformly spaced, i.e., x[i] = x_min + i * dx.
/// Value is linearly interpolated between the two nearest table entries.
/// If x is out of the table range, the nearest table value is returned.
template <class T, class Backend> class UniformTableInterpolator1D {
  Array1DView<T> table_;
  T x_min_, x_max_, dx_, dxi_;

public:
  /// @brief Construct the interpolator from the table and grid parameters.
  UniformTableInterpolator1D(const Array1DView<T> &table, T x_min, T x_max)
      : table_(table), x_min_(x_min), x_max_(x_max),
        dx_((x_max - x_min) / static_cast<T>(table.size() - 1)),
        dxi_(T(1) / dx_) {
    assert(table.size() > 1);
    assert(x_max > x_min);
  }

  /// @brief Interpolate values at x.
  T operator()(T x) const noexcept {
    return interpolate_uniform_table1d(table_.data(), x_min_, dxi_, x,
                                       table_.size());
  }

  /// @brief Interpolate values at the 1D array of x.
  void interpolate(Array1DView<const T> x, Array1DView<T> y) const noexcept {
    assert(x.size() == y.size());
    Range1D range{0, x.size()};
    for_each(
        Backend{}, range, MISO_LAMBDA(int i) {
          y[i] = interpolate_uniform_table1d(table_.data(), x_min_, dxi_, x[i],
                                             table_.size());
        });
  }

  /// @brief Interpolate values at the 3D array of x.
  void interpolate(Array3DView<const T> x, Array3DView<T> y) const noexcept {
    assert(x.extent(0) == y.extent(0));
    assert(x.extent(1) == y.extent(1));
    assert(x.extent(2) == y.extent(2));
    Range1D range{0, x.size()};
    for_each(
        Backend{}, range, MISO_LAMBDA(int i) {
          y[i] = interpolate_uniform_table1d(table_.data(), x_min_, dxi_, x[i],
                                             table_.size());
        });
  }
};

// Interpolate a value at (x0, x1) for 2D table.
// Layout (C row-major; last index contiguous):
//   table[i0 * n1 + i1] corresponds to a[i0][i1]
template <class T>
__device__ __host__ T interpolate_uniform_table2d(const T *table, T x0_min,
                                                  T dx0_inv, T x1_min, T dx1_inv,
                                                  T x0, T x1, int n0,
                                                  int n1) noexcept {
  assert(n0 > 1 && n1 > 1);
  T fi = (x0 - x0_min) * dx0_inv;
  fi = util::clamp<T>(fi, T(0), T(n0 - 1));
  int i = util::clamp<int>(static_cast<int>(fi), 0, n0 - 2);
  T t = fi - T(i);
  T ti = T(1) - t;

  T fj = (x1 - x1_min) * dx1_inv;
  fj = util::clamp<T>(fj, T(0), T(n1 - 1));
  int j = util::clamp<int>(static_cast<int>(fj), 0, n1 - 2);
  T u = fj - T(j);
  T ui = T(1) - u;

  int ij00 = i * n1 + j;
  int ij01 = ij00 + 1;
  int ij10 = ij00 + n1;
  int ij11 = ij10 + 1;
  return ti * ui * table[ij00] + ti * u * table[ij01] + t * ui * table[ij10] +
         t * u * table[ij11];
}

/// @brief Interpolator for 2D (x0, x1) table in the uniform grid.
/// @details The table is assumed to be uniformly spaced,
/// i.e., x0[i] = x0_min + i * dx0, x1[j] = x1_min + j * dx1.
/// Value is linearly interpolated between the two nearest table entries.
/// If (x0, x1) is out of the table range, the nearest table value is returned.
template <class T, class Backend> class UniformTableInterpolator2D {
  Array2DView<T> table_;
  T x0_min_, x0_max_, dx0_, dxi0_;
  T x1_min_, x1_max_, dx1_, dxi1_;

public:
  /// @brief Construct the interpolator from the table and grid parameters.
  UniformTableInterpolator2D(const Array2DView<T> &table, T x0_min, T x0_max,
                             T x1_min, T x1_max)
      : table_(table), x0_min_(x0_min), x0_max_(x0_max), x1_min_(x1_min),
        x1_max_(x1_max),
        dx0_((x0_max - x0_min) / static_cast<T>(table.extent(0) - 1)),
        dxi0_(T(1) / dx0_),
        dx1_((x1_max - x1_min) / static_cast<T>(table.extent(1) - 1)),
        dxi1_(T(1) / dx1_) {
    assert(table.extent(0) > 1);
    assert(table.extent(1) > 1);
    assert(x0_max > x0_min);
    assert(x1_max > x1_min);
  }

  /// @brief Interpolate values at (x0, x1).
  T operator()(T x0, T x1) const noexcept {
    return interpolate_uniform_table2d(table_.data(), x0_min_, dxi0_, x1_min_,
                                       dxi1_, x0, x1, table_.extent(0),
                                       table_.extent(1));
  }

  /// @brief Interpolate values at the 1D arrays of (x0, x1).
  void interpolate(Array1DView<const T> x0, Array1DView<const T> x1,
                   Array1DView<T> y) const noexcept {
    assert(x0.size() == x1.size());
    assert(x0.size() == y.size());
    Range1D range{0, x0.size()};
    for_each(
        Backend{}, range, MISO_LAMBDA(int i) {
          y[i] = interpolate_uniform_table2d(table_.data(), x0_min_, dxi0_,
                                             x1_min_, dxi1_, x0[i], x1[i],
                                             table_.extent(0), table_.extent(1));
        });
  }

  /// @brief Interpolate values at the 3D array of (x0, x1).
  void interpolate(Array3DView<const T> x0, Array3DView<const T> x1,
                   Array3DView<T> y) const noexcept {
    assert(x0.extent(0) == x1.extent(0));
    assert(x0.extent(1) == x1.extent(1));
    assert(x0.extent(2) == x1.extent(2));
    assert(x0.extent(0) == y.extent(0));
    assert(x0.extent(1) == y.extent(1));
    assert(x0.extent(2) == y.extent(2));
    Range1D range{0, x0.size()};
    for_each(
        Backend{}, range, MISO_LAMBDA(int i) {
          y[i] = interpolate_uniform_table2d(table_.data(), x0_min_, dxi0_,
                                             x1_min_, dxi1_, x0[i], x1[i],
                                             table_.extent(0), table_.extent(1));
        });
  }
};

}  // namespace miso
