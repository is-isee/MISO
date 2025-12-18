#pragma once

#include <string>

#include <miso/boundary_condition_core.hpp>
#include <miso/grid_cpu.hpp>

namespace miso {
namespace bnd {

template <typename Real>
void symmetric(Array3D<Real> &arr, const Grid<Real> &grid, Array3D<Real> *fac,
               Real sign, Direction direction, Side side) {
  int i0_, i1_, j0_, j1_, k0_, k1_;
  range_set<Real>(i0_, i1_, j0_, j1_, k0_, k1_, direction, grid);
  for (int i = i0_; i < i1_; ++i) {
    for (int j = j0_; j < j1_; ++j) {
      for (int k = k0_; k < k1_; ++k) {
        int i_ghst = i, i_trgt = i;
        int j_ghst = j, j_trgt = j;
        int k_ghst = k, k_trgt = k;
        switch (direction) {
        case Direction::X:
          symmetric_index<Real>(i, grid.i_total, grid.i_margin, i_ghst, i_trgt,
                                side);
          break;
        case Direction::Y:
          symmetric_index<Real>(j, grid.j_total, grid.j_margin, j_ghst, j_trgt,
                                side);
          break;
        case Direction::Z:
          symmetric_index<Real>(k, grid.k_total, grid.k_margin, k_ghst, k_trgt,
                                side);
          break;
        }
        if (fac) {
          arr(i_ghst, j_ghst, k_ghst) =
              sign * (*fac)(i_trgt, j_trgt, k_trgt) * arr(i_trgt, j_trgt, k_trgt);
        } else {
          arr(i_ghst, j_ghst, k_ghst) = sign * arr(i_trgt, j_trgt, k_trgt);
        }
      }
    }
  }
};

template <typename Real>
void periodic(Array3D<Real> &arr, const Grid<Real> &grid, Direction direction,
              Side side) {
  int i0_, i1_, j0_, j1_, k0_, k1_;
  range_set<Real>(i0_, i1_, j0_, j1_, k0_, k1_, direction, grid);
  for (int i = i0_; i < i1_; ++i) {
    for (int j = j0_; j < j1_; ++j) {
      for (int k = k0_; k < k1_; ++k) {
        int i_ghst = i, i_trgt = i;
        int j_ghst = j, j_trgt = j;
        int k_ghst = k, k_trgt = k;
        switch (direction) {
        case Direction::X:
          periodic_index<Real>(i, grid.i_total, grid.i_margin, i_ghst, i_trgt,
                               side);
          break;
        case Direction::Y:
          periodic_index<Real>(j, grid.j_total, grid.j_margin, j_ghst, j_trgt,
                               side);
          break;
        case Direction::Z:
          periodic_index<Real>(k, grid.k_total, grid.k_margin, k_ghst, k_trgt,
                               side);
          break;
        }
        arr(i_ghst, j_ghst, k_ghst) = arr(i_trgt, j_trgt, k_trgt);
      }
    }
  }
}

}  // namespace bnd
}  // namespace miso
