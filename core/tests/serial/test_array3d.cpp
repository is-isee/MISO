#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#undef USE_CUDA
#include <miso/array3d.hpp>

using namespace miso;

TEST_CASE("Test Array3D CPU" * doctest::test_suite("array3d")) {
  Array3D<int, HostSpace> arr(3, 4, 5);

  // Check dimensions
  REQUIRE(arr.size_x() == 3);
  REQUIRE(arr.size_y() == 4);
  REQUIRE(arr.size_z() == 5);

  // Check access
  arr(1, 2, 3) = 42;
  REQUIRE(arr(1, 2, 3) == 42);
  REQUIRE(arr(1, 2, 3) == arr[1 * (4 * 5) + 2 * 5 + 3]);
}
