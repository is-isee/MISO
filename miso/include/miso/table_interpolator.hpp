#pragma once
#include "array1d.hpp"
#include "execution.hpp"
#include "utility.hpp"

namespace miso {

// Interpolate a value at x.
template <class T>
__device__ __host__ T interpolate_uniform_table(const T *table, T x0, T dxi, T x,
                                                int n) noexcept {
  assert(n > 1);
  T idx = (x - x0) * dxi;
  idx = util::clamp<T>(idx, T(0), static_cast<T>(n - 1));
  int i = util::clamp<int>(static_cast<int>(idx), 0, n - 2);
  T t = idx - static_cast<T>(i);
  return (T(1) - t) * table[i] + t * table[i + 1];
}

/// @brief Interpolator for 1D (x) table in the uniform grid.
/// @details The table is assumed to be sorted in ascending order of x.
/// The table is assumed to be uniformly spaced, i.e., x[i] = x0 + i * dx.
/// Value is linearly interpolated between the two nearest table entries.
/// If x is out of the table range, the nearest table value is returned.
template <class T, class Backend> class UniformTableInterpolator1D {
  Array1DView<T> table_;
  T x0_, x1_, dx_, dxi_;

public:
  /// @brief Construct the interpolator from the table and grid parameters.
  UniformTableInterpolator1D(const Array1DView<T> &table, T x0, T x1)
      : table_(table), x0_(x0), x1_(x1),
        dx_((x1 - x0) / static_cast<T>(table.size() - 1)),
        dxi_(static_cast<T>(table.size() - 1) / (x1 - x0)) {}

  /// @brief Interpolate values at the array of x.
  T interpolate(T x) const noexcept {
    return interpolate_uniform_table(table_.data(), x0_, dxi_, x, table_.size());
  }

  /// @brief Interpolate values at the array of x.
  void interpolate(Array1DView<const T> x, Array1DView<T> y) const noexcept {
    assert(x.size() == y.size());
    Range1D range{0, x.size()};
    for_each(
        Backend{}, range, MISO_LAMBDA(int i) {
          y[i] = interpolate_uniform_table(table_.data(), x0_, dxi_, x[i],
                                           table_.size());
        });
  }
};

// /// @brief 2D table data structure.
// template <class T> struct Table2D {};

// /// @brief Interpolator for 2D (x, y) table in the uniform grid.
// /// @details The table is assumed to be sorted in ascending order of x and y.
// /// The table is assumed to be uniformly spaced, i.e., x[i] = x0 + i * dx, y[j] = y0 + j * dy.
// template <class T> class UniformTableInterpolator2D {};

}  // namespace miso
