#pragma once

#include <miso/cuda_compat.hpp>

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

  DEVICE inline int idx(int i, int j, int k) const {
    return (i * j_total + j) * k_total + k;
  }
};

}  // namespace miso
