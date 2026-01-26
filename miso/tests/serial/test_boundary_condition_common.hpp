#pragma once

#include <cassert>
#include <memory>

#include <doctest/doctest.h>

#include <miso/boundary_condition.hpp>
#include <miso/cuda_compat.hpp>
#include <miso/grid.hpp>
#include <miso/types.hpp>

inline void run_boundary_condition_tests() {
  using namespace miso;
  int i_size = 10, j_size = 11, k_size = 12;
  int margin = 2;
  Real xmin = 0.0, xmax = 1.0, ymin = 0.0, ymax = 2.0, zmin = 0.0, zmax = 3.0;
  Grid<Real, backend::Host> grid(i_size, j_size, k_size, margin, xmin, xmax, ymin,
                                 ymax, zmin, zmax);

  // Test the range_set function
  int i0_, i1_, j0_, j1_, k0_, k1_;

  bnd::range_set<Real>(i0_, i1_, j0_, j1_, k0_, k1_, Direction::X, grid);
  REQUIRE(i0_ == 0);
  REQUIRE(i1_ == grid.i_margin);
  REQUIRE(j0_ == 0);
  REQUIRE(j1_ == grid.j_total);
  REQUIRE(k0_ == 0);
  REQUIRE(k1_ == grid.k_total);

  bnd::range_set<Real>(i0_, i1_, j0_, j1_, k0_, k1_, Direction::Y, grid);
  REQUIRE(i0_ == 0);
  REQUIRE(i1_ == grid.i_total);
  REQUIRE(j0_ == 0);
  REQUIRE(j1_ == grid.j_margin);
  REQUIRE(k0_ == 0);
  REQUIRE(k1_ == grid.k_total);

  bnd::range_set<Real>(i0_, i1_, j0_, j1_, k0_, k1_, Direction::Z, grid);
  REQUIRE(i0_ == 0);
  REQUIRE(i1_ == grid.i_total);
  REQUIRE(j0_ == 0);
  REQUIRE(j1_ == grid.j_total);
  REQUIRE(k0_ == 0);
  REQUIRE(k1_ == grid.k_margin);

  // test for a margin = 2 case
  Array3D<Real, backend::Host> ro(grid.i_total, grid.j_total, grid.k_total);
#ifdef __CUDACC__
  Grid<Real, backend::CUDA> grid_d(grid);
  Array3D<Real, backend::CUDA> ro_d(grid.i_total, grid.j_total, grid.k_total);
#endif  // __CUDACC__

  // set backend and view
#ifdef __CUDACC__
  auto btag = backend::CUDA{};
  auto grid_v = grid_d.const_view();
  auto ro_v = ro_d.view();
#else
  auto btag = backend::Host{};
  auto grid_v = grid.const_view();
  auto ro_v = ro.view();
#endif  // __CUDACC__

  // x boundary test
  for (int i = 0; i < grid.i_total; ++i) {
    for (int j = 0; j < grid.j_total; ++j) {
      for (int k = 0; k < grid.k_total; ++k) {
        ro(i, j, k) = i;
      }
    }
  }

  for (int j = 0; j < grid.j_total; ++j) {
    for (int k = 0; k < grid.k_total; ++k) {
      REQUIRE(ro(0, j, k) != ro(3, j, k));
      REQUIRE(ro(1, j, k) != ro(2, j, k));
      REQUIRE(ro(grid.i_total - 1, j, k) != ro(grid.i_total - 4, j, k));
      REQUIRE(ro(grid.i_total - 2, j, k) != ro(grid.i_total - 3, j, k));
    }
  }

#ifdef __CUDACC__
  ro_d.copy_from(ro);
#endif  // __CUDACC__
  bnd::symmetric<Real>(btag, ro_v, grid_v, Sign::Pos, Direction::X, Side::INNER);
#ifdef __CUDACC__
  ro.copy_from(ro_d);
#endif  // __CUDACC__

  for (int j = 0; j < grid.j_total; ++j) {
    for (int k = 0; k < grid.k_total; ++k) {
      REQUIRE(ro(0, j, k) == ro(3, j, k));
      REQUIRE(ro(1, j, k) == ro(2, j, k));
    }
  }

#ifdef __CUDACC__
  ro_d.copy_from(ro);
#endif  // __CUDACC__
  bnd::symmetric<Real>(btag, ro_v, grid_v, Sign::Neg, Direction::X, Side::INNER);
#ifdef __CUDACC__
  ro.copy_from(ro_d);
#endif  // __CUDACC__

  for (int j = 0; j < grid.j_total; ++j) {
    for (int k = 0; k < grid.k_total; ++k) {
      REQUIRE(ro(0, j, k) == -ro(3, j, k));
      REQUIRE(ro(1, j, k) == -ro(2, j, k));
    }
  }

#ifdef __CUDACC__
  ro_d.copy_from(ro);
#endif  // __CUDACC__
  bnd::symmetric<Real>(btag, ro_v, grid_v, Sign::Pos, Direction::X, Side::OUTER);
#ifdef __CUDACC__
  ro.copy_from(ro_d);
#endif  // __CUDACC__

  for (int j = 0; j < grid.j_total; ++j) {
    for (int k = 0; k < grid.k_total; ++k) {
      REQUIRE(ro(grid.i_total - 1, j, k) == ro(grid.i_total - 4, j, k));
      REQUIRE(ro(grid.i_total - 2, j, k) == ro(grid.i_total - 3, j, k));
    }
  }

#ifdef __CUDACC__
  ro_d.copy_from(ro);
#endif  // __CUDACC__
  bnd::symmetric<Real>(btag, ro_v, grid_v, Sign::Neg, Direction::X, Side::OUTER);
#ifdef __CUDACC__
  ro.copy_from(ro_d);
#endif  // __CUDACC__

  for (int j = 0; j < grid.j_total; ++j) {
    for (int k = 0; k < grid.k_total; ++k) {
      REQUIRE(ro(grid.i_total - 1, j, k) == -ro(grid.i_total - 4, j, k));
      REQUIRE(ro(grid.i_total - 2, j, k) == -ro(grid.i_total - 3, j, k));
    }
  }

  // y boundary test
  for (int i = 0; i < grid.i_total; ++i) {
    for (int j = 0; j < grid.j_total; ++j) {
      for (int k = 0; k < grid.k_total; ++k) {
        ro(i, j, k) = j;
      }
    }
  }

  for (int i = 0; i < grid.i_total; ++i) {
    for (int k = 0; k < grid.k_total; ++k) {
      REQUIRE(ro(i, 0, k) != ro(i, 3, k));
      REQUIRE(ro(i, 1, k) != ro(i, 2, k));
      REQUIRE(ro(i, grid.j_total - 1, k) != ro(i, grid.j_total - 4, k));
      REQUIRE(ro(i, grid.j_total - 2, k) != ro(i, grid.j_total - 3, k));
    }
  }

#ifdef __CUDACC__
  ro_d.copy_from(ro);
#endif  // __CUDACC__
  bnd::symmetric<Real>(btag, ro_v, grid_v, Sign::Pos, Direction::Y, Side::INNER);
#ifdef __CUDACC__
  ro.copy_from(ro_d);
#endif  // __CUDACC__

  for (int i = 0; i < grid.i_total; ++i) {
    for (int k = 0; k < grid.k_total; ++k) {
      REQUIRE(ro(i, 0, k) == ro(i, 3, k));
      REQUIRE(ro(i, 1, k) == ro(i, 2, k));
    }
  }

#ifdef __CUDACC__
  ro_d.copy_from(ro);
#endif  // __CUDACC__
  bnd::symmetric<Real>(btag, ro_v, grid_v, Sign::Neg, Direction::Y, Side::INNER);
#ifdef __CUDACC__
  ro.copy_from(ro_d);
#endif  // __CUDACC__

  for (int i = 0; i < grid.i_total; ++i) {
    for (int k = 0; k < grid.k_total; ++k) {
      REQUIRE(ro(i, 0, k) == -ro(i, 3, k));
      REQUIRE(ro(i, 1, k) == -ro(i, 2, k));
    }
  }

#ifdef __CUDACC__
  ro_d.copy_from(ro);
#endif  // __CUDACC__
  bnd::symmetric<Real>(btag, ro_v, grid_v, Sign::Pos, Direction::Y, Side::OUTER);
#ifdef __CUDACC__
  ro.copy_from(ro_d);
#endif  // __CUDACC__

  for (int i = 0; i < grid.i_total; ++i) {
    for (int k = 0; k < grid.k_total; ++k) {
      REQUIRE(ro(i, grid.j_total - 1, k) == ro(i, grid.j_total - 4, k));
      REQUIRE(ro(i, grid.j_total - 2, k) == ro(i, grid.j_total - 3, k));
    }
  }

#ifdef __CUDACC__
  ro_d.copy_from(ro);
#endif  // __CUDACC__
  bnd::symmetric<Real>(btag, ro_v, grid_v, Sign::Neg, Direction::Y, Side::OUTER);
#ifdef __CUDACC__
  ro.copy_from(ro_d);
#endif  // __CUDACC__

  for (int i = 0; i < grid.i_total; ++i) {
    for (int k = 0; k < grid.k_total; ++k) {
      REQUIRE(ro(i, grid.j_total - 1, k) == -ro(i, grid.j_total - 4, k));
      REQUIRE(ro(i, grid.j_total - 2, k) == -ro(i, grid.j_total - 3, k));
    }
  }

  // z boundary test
  for (int i = 0; i < grid.i_total; ++i) {
    for (int j = 0; j < grid.j_total; ++j) {
      for (int k = 0; k < grid.k_total; ++k) {
        ro(i, j, k) = k;
      }
    }
  }

  for (int i = 0; i < grid.i_total; ++i) {
    for (int j = 0; j < grid.j_total; ++j) {
      REQUIRE(ro(i, j, 0) != ro(i, j, 3));
      REQUIRE(ro(i, j, 1) != ro(i, j, 2));
      REQUIRE(ro(i, j, grid.k_total - 1) != ro(i, j, grid.k_total - 4));
      REQUIRE(ro(i, j, grid.k_total - 2) != ro(i, j, grid.k_total - 3));
    }
  }

#ifdef __CUDACC__
  ro_d.copy_from(ro);
#endif  // __CUDACC__
  bnd::symmetric<Real>(btag, ro_v, grid_v, Sign::Pos, Direction::Z, Side::INNER);
#ifdef __CUDACC__
  ro.copy_from(ro_d);
#endif  // __CUDACC__

  for (int i = 0; i < grid.i_total; ++i) {
    for (int j = 0; j < grid.j_total; ++j) {
      REQUIRE(ro(i, j, 0) == ro(i, j, 3));
      REQUIRE(ro(i, j, 1) == ro(i, j, 2));
    }
  }

#ifdef __CUDACC__
  ro_d.copy_from(ro);
#endif  // __CUDACC__
  bnd::symmetric<Real>(btag, ro_v, grid_v, Sign::Neg, Direction::Z, Side::INNER);
#ifdef __CUDACC__
  ro.copy_from(ro_d);
#endif  // __CUDACC__

  for (int i = 0; i < grid.i_total; ++i) {
    for (int j = 0; j < grid.j_total; ++j) {
      REQUIRE(ro(i, j, 0) == -ro(i, j, 3));
      REQUIRE(ro(i, j, 1) == -ro(i, j, 2));
    }
  }

#ifdef __CUDACC__
  ro_d.copy_from(ro);
#endif  // __CUDACC__
  bnd::symmetric<Real>(btag, ro_v, grid_v, Sign::Pos, Direction::Z, Side::OUTER);
#ifdef __CUDACC__
  ro.copy_from(ro_d);
#endif  // __CUDACC__

  for (int i = 0; i < grid.i_total; ++i) {
    for (int j = 0; j < grid.j_total; ++j) {
      REQUIRE(ro(i, j, grid.k_total - 1) == ro(i, j, grid.k_total - 4));
      REQUIRE(ro(i, j, grid.k_total - 2) == ro(i, j, grid.k_total - 3));
    }
  }

#ifdef __CUDACC__
  ro_d.copy_from(ro);
#endif  // __CUDACC__
  bnd::symmetric<Real>(btag, ro_v, grid_v, Sign::Neg, Direction::Z, Side::OUTER);
#ifdef __CUDACC__
  ro.copy_from(ro_d);
#endif  // __CUDACC__

  for (int i = 0; i < grid.i_total; ++i) {
    for (int j = 0; j < grid.j_total; ++j) {

      REQUIRE(ro(i, j, grid.k_total - 1) == -ro(i, j, grid.k_total - 4));
      REQUIRE(ro(i, j, grid.k_total - 2) == -ro(i, j, grid.k_total - 3));
    }
  }

  int i;
  int i_total = 10;
  int i_margin = 2;
  int i_ghst, i_trgt;

  i = 0;
  bnd::symmetric_index<Real>(i, i_total, i_margin, i_ghst, i_trgt, Side::INNER);
  REQUIRE(i_ghst == 0);
  REQUIRE(i_trgt == 3);

  bnd::symmetric_index<Real>(i, i_total, i_margin, i_ghst, i_trgt, Side::OUTER);
  REQUIRE(i_ghst == 8);
  REQUIRE(i_trgt == 7);

  i = 1;
  bnd::symmetric_index<Real>(i, i_total, i_margin, i_ghst, i_trgt, Side::INNER);
  REQUIRE(i_ghst == 1);
  REQUIRE(i_trgt == 2);

  bnd::symmetric_index<Real>(i, i_total, i_margin, i_ghst, i_trgt, Side::OUTER);
  REQUIRE(i_ghst == 9);
  REQUIRE(i_trgt == 6);
}
