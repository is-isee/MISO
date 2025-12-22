#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <cassert>
#include <doctest/doctest.h>

#undef USE_CUDA
#include <miso/array3d_cpu.hpp>

TEST_CASE("Test Array3D CPU" * doctest::test_suite("array3d")) {
  miso::Array3D<double> arr(3, 4, 5);

  // Check dimensions
  REQUIRE(arr.size_x() == 3);
  REQUIRE(arr.size_y() == 4);
  REQUIRE(arr.size_z() == 5);

  // Check access
  arr(1, 2, 3) = 42.0;
  REQUIRE(arr(1, 2, 3) == 42.0);
}
