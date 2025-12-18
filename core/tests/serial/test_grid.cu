#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <cassert>
#include <doctest/doctest.h>

#include <miso/grid.hpp>

__global__ void test_grid_kernel(miso::GridDevice<double> grid_d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < grid_d.i_total && j < grid_d.j_total && k < grid_d.k_total) {
    grid_d.x[i] = 1.0;  // Set all x elements to 1.0
    grid_d.y[j] = 2.0;  // Set all y elements to 2.0
    grid_d.z[k] = 3.0;  // Set all z elements to 3.0
  }
}

TEST_CASE("Test Grid GPU" * doctest::test_suite("grid")) {
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

  miso::Grid<double> grid(i_size, j_size, k_size, margin, xmin, xmax, ymin, ymax,
                          zmin, zmax);
  grid.mask.allocate(grid.i_total, grid.j_total, grid.k_total);
  miso::GridDevice<double> grid_d(grid);

  grid_d.copy_from_host(grid);

  test_grid_kernel<<<dim3(1, 1, 1), dim3(i_size, j_size, k_size)>>>(grid_d);
  grid_d.copy_to_host(grid);

  for (int i = 0; i < i_size; ++i) {
    REQUIRE(grid.x[i] == 1.0);
  }
  for (int j = 0; j < j_size; ++j) {
    REQUIRE(grid.y[j] == 2.0);
  }
  for (int k = 0; k < k_size; ++k) {
    REQUIRE(grid.z[k] == 3.0);
  }
}
