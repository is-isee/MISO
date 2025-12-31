#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#undef USE_CUDA
#include <miso/grid.hpp>

using namespace miso;

TEST_CASE("Test Grid CPU" * doctest::test_suite("grid")) {
  // Test the Grid constructor and accessors
  int i_size = 3;
  int j_size = 4;
  int k_size = 5;
  int margin = 1;
  double xmin = 0.0;
  double xmax = 1.0;
  double ymin = 2.0;
  double ymax = 3.0;
  double zmin = 4.0;
  double zmax = 5.0;

  Grid<double, backend::Host> grid(i_size, j_size, k_size, margin, xmin, xmax, ymin,
                               ymax, zmin, zmax);

  // Check dimensions
  REQUIRE(grid.i_size == i_size);
  REQUIRE(grid.j_size == j_size);
  REQUIRE(grid.k_size == k_size);
  REQUIRE(grid.margin == margin);
  REQUIRE(grid.i_total == i_size + 2 * margin);
  REQUIRE(grid.j_total == j_size + 2 * margin);
  REQUIRE(grid.k_total == k_size + 2 * margin);

  // Check coordinates
  REQUIRE(grid.x.size() == grid.i_total);
  REQUIRE(grid.y.size() == grid.j_total);
  REQUIRE(grid.z.size() == grid.k_total);

  // Check dx values
  for (int i = grid.i_margin; i < grid.i_total - grid.margin; ++i) {
    REQUIRE(grid.dx[i] > 0);
    REQUIRE(grid.x[i] >= grid.xmin);
    REQUIRE(grid.x[i] <= grid.xmax);
  }
}
