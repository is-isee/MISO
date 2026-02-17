#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <miso/grid.hpp>

using namespace miso;

TEST_CASE("Test Grid CPU" * doctest::test_suite("grid")) {
  // Test the Grid constructor and accessors
  int i_size = 3;
  int j_size = 4;
  int k_size = 5;
  int margin = 1;
  double x_min = 0.0;
  double x_max = 1.0;
  double y_min = 2.0;
  double y_max = 3.0;
  double z_min = 4.0;
  double z_max = 5.0;

  Grid<double, backend::Host> grid(i_size, j_size, k_size, margin, x_min, x_max,
                                   y_min, y_max, z_min, z_max);

  // Check dimensions
  REQUIRE(grid.i_size == i_size);
  REQUIRE(grid.j_size == j_size);
  REQUIRE(grid.k_size == k_size);
  REQUIRE(grid.i_margin == margin);
  REQUIRE(grid.j_margin == margin);
  REQUIRE(grid.k_margin == margin);
  REQUIRE(grid.i_total == i_size + 2 * margin);
  REQUIRE(grid.j_total == j_size + 2 * margin);
  REQUIRE(grid.k_total == k_size + 2 * margin);

  // Check coordinates
  REQUIRE(grid.x.size() == grid.i_total);
  REQUIRE(grid.y.size() == grid.j_total);
  REQUIRE(grid.z.size() == grid.k_total);

  // Check dx values
  for (int i = grid.i_margin; i < grid.i_total - grid.i_margin; ++i) {
    REQUIRE(grid.dx[i] > 0);
    REQUIRE(grid.x[i] >= grid.x_min);
    REQUIRE(grid.x[i] <= grid.x_max);
  }
}
