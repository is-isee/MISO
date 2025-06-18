#include <doctest/doctest.h>
#include <cassert>
#include <memory>
#include "array3d_cpu.hpp"
#include "grid_cpu.hpp"
#include "boundary_condition_core.hpp"
#ifdef USE_CUDA
#include "boundary_condition_core_gpu.cuh"
#else
#include "boundary_condition_core_cpu.hpp"
#endif
#include "boundary_condition_base.hpp"
#include "types.hpp"


TEST_CASE("Test BoundaryCondition class") {
    int i_size = 10, j_size = 11, k_size = 12;
    int margin = 2;
    Real xmin = 0.0, xmax = 1.0, ymin = 0.0, ymax = 2.0, zmin = 0.0, zmax = 3.0;

    Grid<Real> grid(i_size, j_size, k_size, margin, xmin, xmax, ymin, ymax, zmin, zmax);

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
    Array3D<Real> arr(grid.i_total, grid.j_total, grid.k_total);

    // x boundary test
    for (int i = 0; i < grid.i_total; ++i) {
        for (int j = 0; j < grid.j_total; ++j) {
            for (int k = 0; k < grid.k_total; ++k) {
                arr(i, j, k) = i;
            }
        }
    }

    for (int j = 0; j < grid.j_total; ++j) {
        for (int k = 0; k < grid.k_total; ++k) {
            REQUIRE(arr(0, j, k) != arr(3, j, k));
            REQUIRE(arr(1, j, k) != arr(2, j, k));            
            REQUIRE(arr(grid.i_total - 1, j, k) != arr(grid.i_total-4, j, k));
            REQUIRE(arr(grid.i_total - 2, j, k) != arr(grid.i_total-3, j, k));
        }
    }

    bnd::symmetric<Real>(arr, grid, nullptr, 1.0, bnd::Direction::X, bnd::Side::INNER);
    for (int j = 0; j < grid.j_total; ++j) {
        for (int k = 0; k < grid.k_total; ++k) {
            REQUIRE(arr(0, j, k) == arr(3, j, k));
            REQUIRE(arr(1, j, k) == arr(2, j, k));            
        }
    }

    bnd::symmetric<Real>(arr, grid, nullptr, -1.0, bnd::Direction::X, bnd::Side::INNER);
    for (int j = 0; j < grid.j_total; ++j) {
        for (int k = 0; k < grid.k_total; ++k) {
            REQUIRE(arr(0, j, k) == -arr(3, j, k));
            REQUIRE(arr(1, j, k) == -arr(2, j, k));            
        }
    }

    bnd::symmetric<Real>(arr, grid, nullptr, 1.0, bnd::Direction::X, bnd::Side::OUTER);
    for (int j = 0; j < grid.j_total; ++j) {
        for (int k = 0; k < grid.k_total; ++k) {
            REQUIRE(arr(grid.i_total - 1, j, k) == arr(grid.i_total-4, j, k));
            REQUIRE(arr(grid.i_total - 2, j, k) == arr(grid.i_total-3, j, k));
        }
    }

    bnd::symmetric<Real>(arr, grid, nullptr, -1.0, bnd::Direction::X, bnd::Side::OUTER);
    for (int j = 0; j < grid.j_total; ++j) {
        for (int k = 0; k < grid.k_total; ++k) {
            REQUIRE(arr(grid.i_total - 1, j, k) == -arr(grid.i_total-4, j, k));
            REQUIRE(arr(grid.i_total - 2, j, k) == -arr(grid.i_total-3, j, k));
        }
    }

    // y boundary test
    for (int i = 0; i < grid.i_total; ++i) {
        for (int j = 0; j < grid.j_total; ++j) {
            for (int k = 0; k < grid.k_total; ++k) {
                arr(i, j, k) = j;
            }
        }
    }

    for (int i = 0; i < grid.i_total; ++i) {
        for (int k = 0; k < grid.k_total; ++k) {
            REQUIRE(arr(i, 0, k) != arr(i, 3, k));
            REQUIRE(arr(i, 1, k) != arr(i, 2, k));            
            REQUIRE(arr(i, grid.j_total - 1, k) != arr(i, grid.j_total-4, k));
            REQUIRE(arr(i, grid.j_total - 2, k) != arr(i, grid.j_total-3, k));
        }
    }

    bnd::symmetric<Real>(arr, grid, nullptr, 1.0, bnd::Direction::Y, bnd::Side::INNER);
    for (int i = 0; i < grid.i_total; ++i) {
        for (int k = 0; k < grid.k_total; ++k) {
            REQUIRE(arr(i, 0, k) == arr(i, 3, k));
            REQUIRE(arr(i, 1, k) == arr(i, 2, k));            
        }
    }

    bnd::symmetric<Real>(arr, grid, nullptr, -1.0, bnd::Direction::Y, bnd::Side::INNER);
    for (int i = 0; i < grid.i_total; ++i) {
        for (int k = 0; k < grid.k_total; ++k) {
            REQUIRE(arr(i, 0, k) == -arr(i, 3, k));
            REQUIRE(arr(i, 1, k) == -arr(i, 2, k));
        }
    }

    bnd::symmetric<Real>(arr, grid, nullptr, 1.0, bnd::Direction::Y, bnd::Side::OUTER);
    for (int i = 0; i < grid.i_total; ++i) {
        for (int k = 0; k < grid.k_total; ++k) {
            REQUIRE(arr(i, grid.j_total - 1, k) == arr(i, grid.j_total-4, k));
            REQUIRE(arr(i, grid.j_total - 2, k) == arr(i, grid.j_total-3, k));
        }
    }

    bnd::symmetric<Real>(arr, grid, nullptr, -1.0, bnd::Direction::Y, bnd::Side::OUTER);
    for (int i = 0; i < grid.i_total; ++i) {
        for (int k = 0; k < grid.k_total; ++k) {
            REQUIRE(arr(i, grid.j_total - 1, k) == -arr(i, grid.j_total - 4, k));
            REQUIRE(arr(i, grid.j_total - 2, k) == -arr(i, grid.j_total - 3, k));
        }
    }

    // z boundary test
    for (int i = 0; i < grid.i_total; ++i) {
        for (int j = 0; j < grid.j_total; ++j) {
            for (int k = 0; k < grid.k_total; ++k) {
                arr(i, j, k) = k;
            }
        }
    }

    for (int i = 0; i < grid.i_total; ++i) {
        for (int j = 0; j < grid.j_total; ++j) {
            REQUIRE(arr(i, j, 0) != arr(i, j, 3));
            REQUIRE(arr(i, j, 1) != arr(i, j, 2));
            REQUIRE(arr(i, j, grid.k_total - 1) != arr(i, j, grid.k_total-4));
            REQUIRE(arr(i, j, grid.k_total - 2) != arr(i, j, grid.k_total-3));
        }
    }

    bnd::symmetric<Real>(arr, grid, nullptr, 1.0, bnd::Direction::Z, bnd::Side::INNER);
    for (int i = 0; i < grid.i_total; ++i) {
        for (int j = 0; j < grid.j_total; ++j) {
            REQUIRE(arr(i, j, 0) == arr(i, j, 3));
            REQUIRE(arr(i, j, 1) == arr(i, j, 2));            
        }
    }

    bnd::symmetric<Real>(arr, grid, nullptr, -1.0, bnd::Direction::Z, bnd::Side::INNER);
    for (int i = 0; i < grid.i_total; ++i) {
        for (int j = 0; j < grid.j_total; ++j) {
            REQUIRE(arr(i, j, 0) == -arr(i, j, 3));
            REQUIRE(arr(i, j, 1) == -arr(i, j, 2));
        }
    }

    bnd::symmetric<Real>(arr, grid, nullptr, 1.0, bnd::Direction::Z, bnd::Side::OUTER);
    for (int i = 0; i < grid.i_total; ++i) {
        for (int j = 0; j < grid.j_total; ++j) {
            REQUIRE(arr(i, j, grid.k_total - 1) == arr(i, j, grid.k_total-4));
            REQUIRE(arr(i, j, grid.k_total - 2) == arr(i, j, grid.k_total-3));
        }
    }

    bnd::symmetric<Real>(arr, grid, nullptr, -1.0, bnd::Direction::Z, bnd::Side::OUTER);
    for (int i = 0; i < grid.i_total; ++i) {
        for (int j = 0; j < grid.j_total; ++j) {

            REQUIRE(arr(i, j, grid.k_total - 1) == -arr(i, j, grid.k_total - 4));
            REQUIRE(arr(i, j, grid.k_total - 2) == -arr(i, j, grid.k_total - 3));
        }
    }
    
    int i;
    int i_total = 10;
    int i_margin = 2;
    int i_ghst, i_trgt;

    i = 0;
    bnd::symmetric_index<Real>(i, i_total, i_margin, i_ghst, i_trgt, bnd::Side::INNER);
    REQUIRE(i_ghst == 0);
    REQUIRE(i_trgt == 3);

    bnd::symmetric_index<Real>(i, i_total, i_margin, i_ghst, i_trgt, bnd::Side::OUTER);
    REQUIRE(i_ghst == 8);
    REQUIRE(i_trgt == 7);

    i = 1;
    bnd::symmetric_index<Real>(i, i_total, i_margin, i_ghst, i_trgt, bnd::Side::INNER);
    REQUIRE(i_ghst == 1);
    REQUIRE(i_trgt == 2);

    bnd::symmetric_index<Real>(i, i_total, i_margin, i_ghst, i_trgt, bnd::Side::OUTER);
    REQUIRE(i_ghst == 9);
    REQUIRE(i_trgt == 6);
}
