#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <miso/array1d.hpp>

using namespace miso;

TEST_CASE("Test Array1D CPU" * doctest::test_suite("array1d")) {
  Array1D<int, backend::Host> arr(3);

  // Check dimensions
  REQUIRE(arr.size() == 3);

  // Check access
  auto view = arr.view();
  view[1] = 42;
  REQUIRE(arr[1] == 42);
}
