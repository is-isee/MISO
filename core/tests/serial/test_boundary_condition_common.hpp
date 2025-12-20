#pragma once
#include <cassert>
#include <doctest/doctest.h>
#include <memory>

#include <miso/boundary_condition.hpp>
#include <miso/cuda_compat.hpp>
#include <miso/grid.hpp>
#include <miso/mhd.hpp>
#include <miso/types.hpp>

inline void run_boundary_condition_tests() {
  using namespace miso;
  int i_size = 10, j_size = 11, k_size = 12;
  int margin = 2;
  Real xmin = 0.0, xmax = 1.0, ymin = 0.0, ymax = 2.0, zmin = 0.0, zmax = 3.0;
  Grid<Real> grid(i_size, j_size, k_size, margin, xmin, xmax, ymin, ymax, zmin,
                  zmax);

  // Test the range_set function
  int i0_, i1_, j0_, j1_, k0_, k1_;

  bnd::range_set<Real>(i0_, i1_, j0_, j1_, k0_, k1_, bnd::Direction::X, grid);
  REQUIRE(i0_ == 0);
  REQUIRE(i1_ == grid.i_margin);
  REQUIRE(j0_ == 0);
  REQUIRE(j1_ == grid.j_total);
  REQUIRE(k0_ == 0);
  REQUIRE(k1_ == grid.k_total);

  bnd::range_set<Real>(i0_, i1_, j0_, j1_, k0_, k1_, bnd::Direction::Y, grid);
  REQUIRE(i0_ == 0);
  REQUIRE(i1_ == grid.i_total);
  REQUIRE(j0_ == 0);
  REQUIRE(j1_ == grid.j_margin);
  REQUIRE(k0_ == 0);
  REQUIRE(k1_ == grid.k_total);

  bnd::range_set<Real>(i0_, i1_, j0_, j1_, k0_, k1_, bnd::Direction::Z, grid);
  REQUIRE(i0_ == 0);
  REQUIRE(i1_ == grid.i_total);
  REQUIRE(j0_ == 0);
  REQUIRE(j1_ == grid.j_total);
  REQUIRE(k0_ == 0);
  REQUIRE(k1_ == grid.k_margin);

  // test for a margin = 2 case
  mhd::MHDCore<Real> qq(grid.i_total, grid.j_total, grid.k_total);
#ifdef USE_CUDA
  mhd::MHDStreams cuda_streams;
  GridDevice<Real> grid_d(grid);
  mhd::MHDCoreDevice<Real> qq_d(grid);
#endif

  // x boundary test
  for (int i = 0; i < grid.i_total; ++i) {
    for (int j = 0; j < grid.j_total; ++j) {
      for (int k = 0; k < grid.k_total; ++k) {
        qq.ro(i, j, k) = i;
      }
    }
  }

  for (int j = 0; j < grid.j_total; ++j) {
    for (int k = 0; k < grid.k_total; ++k) {
      REQUIRE(qq.ro(0, j, k) != qq.ro(3, j, k));
      REQUIRE(qq.ro(1, j, k) != qq.ro(2, j, k));
      REQUIRE(qq.ro(grid.i_total - 1, j, k) != qq.ro(grid.i_total - 4, j, k));
      REQUIRE(qq.ro(grid.i_total - 2, j, k) != qq.ro(grid.i_total - 3, j, k));
    }
  }

#ifdef USE_CUDA
  qq_d.copy_from_host(qq, cuda_streams);
  bnd::symmetric<Real>(qq_d.ro, grid_d, qq_d.ro, 1.0, bnd::Direction::X,
                       bnd::Side::INNER);
  qq_d.copy_to_host(qq, cuda_streams);
#else
  bnd::symmetric<Real>(qq.ro, grid, nullptr, 1.0, bnd::Direction::X,
                       bnd::Side::INNER);
#endif

  for (int j = 0; j < grid.j_total; ++j) {
    for (int k = 0; k < grid.k_total; ++k) {
      REQUIRE(qq.ro(0, j, k) == qq.ro(3, j, k));
      REQUIRE(qq.ro(1, j, k) == qq.ro(2, j, k));
    }
  }

#ifdef USE_CUDA
  qq_d.copy_from_host(qq, cuda_streams);
  bnd::symmetric<Real>(qq_d.ro, grid, qq_d.ro, -1.0, bnd::Direction::X,
                       bnd::Side::INNER);
  qq_d.copy_to_host(qq, cuda_streams);
#else
  bnd::symmetric<Real>(qq.ro, grid, nullptr, -1.0, bnd::Direction::X,
                       bnd::Side::INNER);
#endif

  for (int j = 0; j < grid.j_total; ++j) {
    for (int k = 0; k < grid.k_total; ++k) {
      REQUIRE(qq.ro(0, j, k) == -qq.ro(3, j, k));
      REQUIRE(qq.ro(1, j, k) == -qq.ro(2, j, k));
    }
  }

#ifdef USE_CUDA
  qq_d.copy_from_host(qq, cuda_streams);
  bnd::symmetric<Real>(qq_d.ro, grid, qq_d.ro, 1.0, bnd::Direction::X,
                       bnd::Side::OUTER);
  qq_d.copy_to_host(qq, cuda_streams);
#else
  bnd::symmetric<Real>(qq.ro, grid, nullptr, 1.0, bnd::Direction::X,
                       bnd::Side::OUTER);
#endif

  for (int j = 0; j < grid.j_total; ++j) {
    for (int k = 0; k < grid.k_total; ++k) {
      REQUIRE(qq.ro(grid.i_total - 1, j, k) == qq.ro(grid.i_total - 4, j, k));
      REQUIRE(qq.ro(grid.i_total - 2, j, k) == qq.ro(grid.i_total - 3, j, k));
    }
  }

#ifdef USE_CUDA
  qq_d.copy_from_host(qq, cuda_streams);
  bnd::symmetric<Real>(qq_d.ro, grid_d, qq_d.ro, -1.0, bnd::Direction::X,
                       bnd::Side::OUTER);
  qq_d.copy_to_host(qq, cuda_streams);
#else
  bnd::symmetric<Real>(qq.ro, grid, nullptr, -1.0, bnd::Direction::X,
                       bnd::Side::OUTER);
#endif

  for (int j = 0; j < grid.j_total; ++j) {
    for (int k = 0; k < grid.k_total; ++k) {
      REQUIRE(qq.ro(grid.i_total - 1, j, k) == -qq.ro(grid.i_total - 4, j, k));
      REQUIRE(qq.ro(grid.i_total - 2, j, k) == -qq.ro(grid.i_total - 3, j, k));
    }
  }

  // y boundary test
  for (int i = 0; i < grid.i_total; ++i) {
    for (int j = 0; j < grid.j_total; ++j) {
      for (int k = 0; k < grid.k_total; ++k) {
        qq.ro(i, j, k) = j;
      }
    }
  }

  for (int i = 0; i < grid.i_total; ++i) {
    for (int k = 0; k < grid.k_total; ++k) {
      REQUIRE(qq.ro(i, 0, k) != qq.ro(i, 3, k));
      REQUIRE(qq.ro(i, 1, k) != qq.ro(i, 2, k));
      REQUIRE(qq.ro(i, grid.j_total - 1, k) != qq.ro(i, grid.j_total - 4, k));
      REQUIRE(qq.ro(i, grid.j_total - 2, k) != qq.ro(i, grid.j_total - 3, k));
    }
  }

#ifdef USE_CUDA
  qq_d.copy_from_host(qq, cuda_streams);
  bnd::symmetric<Real>(qq_d.ro, grid_d, qq_d.ro, 1.0, bnd::Direction::Y,
                       bnd::Side::INNER);
  qq_d.copy_to_host(qq, cuda_streams);
#else
  bnd::symmetric<Real>(qq.ro, grid, nullptr, 1.0, bnd::Direction::Y,
                       bnd::Side::INNER);
#endif

  for (int i = 0; i < grid.i_total; ++i) {
    for (int k = 0; k < grid.k_total; ++k) {
      REQUIRE(qq.ro(i, 0, k) == qq.ro(i, 3, k));
      REQUIRE(qq.ro(i, 1, k) == qq.ro(i, 2, k));
    }
  }

#ifdef USE_CUDA
  qq_d.copy_from_host(qq, cuda_streams);
  bnd::symmetric<Real>(qq_d.ro, grid_d, qq_d.ro, -1.0, bnd::Direction::Y,
                       bnd::Side::INNER);
  qq_d.copy_to_host(qq, cuda_streams);
#else
  bnd::symmetric<Real>(qq.ro, grid, nullptr, -1.0, bnd::Direction::Y,
                       bnd::Side::INNER);
#endif

  for (int i = 0; i < grid.i_total; ++i) {
    for (int k = 0; k < grid.k_total; ++k) {
      REQUIRE(qq.ro(i, 0, k) == -qq.ro(i, 3, k));
      REQUIRE(qq.ro(i, 1, k) == -qq.ro(i, 2, k));
    }
  }

#ifdef USE_CUDA
  qq_d.copy_from_host(qq, cuda_streams);
  bnd::symmetric<Real>(qq_d.ro, grid_d, qq_d.ro, 1.0, bnd::Direction::Y,
                       bnd::Side::OUTER);
  qq_d.copy_to_host(qq, cuda_streams);
#else
  bnd::symmetric<Real>(qq.ro, grid, nullptr, 1.0, bnd::Direction::Y,
                       bnd::Side::OUTER);
#endif

  for (int i = 0; i < grid.i_total; ++i) {
    for (int k = 0; k < grid.k_total; ++k) {
      REQUIRE(qq.ro(i, grid.j_total - 1, k) == qq.ro(i, grid.j_total - 4, k));
      REQUIRE(qq.ro(i, grid.j_total - 2, k) == qq.ro(i, grid.j_total - 3, k));
    }
  }

#ifdef USE_CUDA
  qq_d.copy_from_host(qq, cuda_streams);
  bnd::symmetric<Real>(qq_d.ro, grid_d, qq_d.ro, -1.0, bnd::Direction::Y,
                       bnd::Side::OUTER);
  qq_d.copy_to_host(qq, cuda_streams);
#else
  bnd::symmetric<Real>(qq.ro, grid, nullptr, -1.0, bnd::Direction::Y,
                       bnd::Side::OUTER);
#endif

  for (int i = 0; i < grid.i_total; ++i) {
    for (int k = 0; k < grid.k_total; ++k) {
      REQUIRE(qq.ro(i, grid.j_total - 1, k) == -qq.ro(i, grid.j_total - 4, k));
      REQUIRE(qq.ro(i, grid.j_total - 2, k) == -qq.ro(i, grid.j_total - 3, k));
    }
  }

  // z boundary test
  for (int i = 0; i < grid.i_total; ++i) {
    for (int j = 0; j < grid.j_total; ++j) {
      for (int k = 0; k < grid.k_total; ++k) {
        qq.ro(i, j, k) = k;
      }
    }
  }

  for (int i = 0; i < grid.i_total; ++i) {
    for (int j = 0; j < grid.j_total; ++j) {
      REQUIRE(qq.ro(i, j, 0) != qq.ro(i, j, 3));
      REQUIRE(qq.ro(i, j, 1) != qq.ro(i, j, 2));
      REQUIRE(qq.ro(i, j, grid.k_total - 1) != qq.ro(i, j, grid.k_total - 4));
      REQUIRE(qq.ro(i, j, grid.k_total - 2) != qq.ro(i, j, grid.k_total - 3));
    }
  }

#ifdef USE_CUDA
  qq_d.copy_from_host(qq, cuda_streams);
  bnd::symmetric<Real>(qq_d.ro, grid_d, qq_d.ro, 1.0, bnd::Direction::Z,
                       bnd::Side::INNER);
  qq_d.copy_to_host(qq, cuda_streams);
#else
  bnd::symmetric<Real>(qq.ro, grid, nullptr, 1.0, bnd::Direction::Z,
                       bnd::Side::INNER);
#endif

  for (int i = 0; i < grid.i_total; ++i) {
    for (int j = 0; j < grid.j_total; ++j) {
      REQUIRE(qq.ro(i, j, 0) == qq.ro(i, j, 3));
      REQUIRE(qq.ro(i, j, 1) == qq.ro(i, j, 2));
    }
  }

#ifdef USE_CUDA
  qq_d.copy_from_host(qq, cuda_streams);
  bnd::symmetric<Real>(qq_d.ro, grid_d, qq_d.ro, -1.0, bnd::Direction::Z,
                       bnd::Side::INNER);
  qq_d.copy_to_host(qq, cuda_streams);
#else
  bnd::symmetric<Real>(qq.ro, grid, nullptr, -1.0, bnd::Direction::Z,
                       bnd::Side::INNER);
#endif

  for (int i = 0; i < grid.i_total; ++i) {
    for (int j = 0; j < grid.j_total; ++j) {
      REQUIRE(qq.ro(i, j, 0) == -qq.ro(i, j, 3));
      REQUIRE(qq.ro(i, j, 1) == -qq.ro(i, j, 2));
    }
  }

#ifdef USE_CUDA
  qq_d.copy_from_host(qq, cuda_streams);
  bnd::symmetric<Real>(qq_d.ro, grid_d, qq_d.ro, 1.0, bnd::Direction::Z,
                       bnd::Side::OUTER);
  qq_d.copy_to_host(qq, cuda_streams);
#else
  bnd::symmetric<Real>(qq.ro, grid, nullptr, 1.0, bnd::Direction::Z,
                       bnd::Side::OUTER);
#endif

  for (int i = 0; i < grid.i_total; ++i) {
    for (int j = 0; j < grid.j_total; ++j) {
      REQUIRE(qq.ro(i, j, grid.k_total - 1) == qq.ro(i, j, grid.k_total - 4));
      REQUIRE(qq.ro(i, j, grid.k_total - 2) == qq.ro(i, j, grid.k_total - 3));
    }
  }

#ifdef USE_CUDA
  qq_d.copy_from_host(qq, cuda_streams);
  bnd::symmetric<Real>(qq_d.ro, grid_d, qq_d.ro, -1.0, bnd::Direction::Z,
                       bnd::Side::OUTER);
  qq_d.copy_to_host(qq, cuda_streams);
#else
  bnd::symmetric<Real>(qq.ro, grid, nullptr, -1.0, bnd::Direction::Z,
                       bnd::Side::OUTER);
#endif

  for (int i = 0; i < grid.i_total; ++i) {
    for (int j = 0; j < grid.j_total; ++j) {

      REQUIRE(qq.ro(i, j, grid.k_total - 1) == -qq.ro(i, j, grid.k_total - 4));
      REQUIRE(qq.ro(i, j, grid.k_total - 2) == -qq.ro(i, j, grid.k_total - 3));
    }
  }

  int i;
  int i_total = 10;
  int i_margin = 2;
  int i_ghst, i_trgt;

  i = 0;
  bnd::symmetric_index<Real>(i, i_total, i_margin, i_ghst, i_trgt,
                             bnd::Side::INNER);
  REQUIRE(i_ghst == 0);
  REQUIRE(i_trgt == 3);

  bnd::symmetric_index<Real>(i, i_total, i_margin, i_ghst, i_trgt,
                             bnd::Side::OUTER);
  REQUIRE(i_ghst == 8);
  REQUIRE(i_trgt == 7);

  i = 1;
  bnd::symmetric_index<Real>(i, i_total, i_margin, i_ghst, i_trgt,
                             bnd::Side::INNER);
  REQUIRE(i_ghst == 1);
  REQUIRE(i_trgt == 2);

  bnd::symmetric_index<Real>(i, i_total, i_margin, i_ghst, i_trgt,
                             bnd::Side::OUTER);
  REQUIRE(i_ghst == 9);
  REQUIRE(i_trgt == 6);
}
