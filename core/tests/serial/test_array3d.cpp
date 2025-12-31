#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <miso/array3d.hpp>

using namespace miso;

TEST_CASE("Test Array3D CPU" * doctest::test_suite("array3d")) {
  Array3D<int, backend::Host> arr(3, 4, 5);

  // Check dimensions
  REQUIRE(arr.extent(0) == 3);
  REQUIRE(arr.extent(1) == 4);
  REQUIRE(arr.extent(2) == 5);
  REQUIRE(arr.shape() == std::array<int, 3>{3, 4, 5});
  REQUIRE(arr.size() == 3 * 4 * 5);

  // Check access
  arr(1, 2, 3) = 42;
  REQUIRE(arr(1, 2, 3) == 42);
  REQUIRE(arr(1, 2, 3) == arr[(1 * 4 + 2) * 5 + 3]);
}
