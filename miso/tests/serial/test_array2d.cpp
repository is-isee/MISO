#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <miso/array2d.hpp>

using namespace miso;

TEST_CASE("Test Array2D CPU" * doctest::test_suite("array2d")) {
  Array2D<int, backend::Host> arr(3, 4);

  // Check dimensions
  REQUIRE(arr.extent(0) == 3);
  REQUIRE(arr.extent(1) == 4);
  REQUIRE(arr.shape() == std::array<int, 2>{3, 4});
  REQUIRE(arr.size() == 3 * 4);

  // Check access
  auto view = arr.view();
  view(1, 2) = 42;
  REQUIRE(arr(1, 2) == 42);
  REQUIRE(&view(1, 2) == &arr(1, 2));
  REQUIRE(view(1, 2) == arr[(1 * 4 + 2)]);
}
