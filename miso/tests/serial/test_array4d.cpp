#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <miso/array4d.hpp>

using namespace miso;

TEST_CASE("Test Array4D CPU" * doctest::test_suite("array4d")) {
  Array4D<int, backend::Host> arr(2, 3, 4, 5);

  REQUIRE(arr.extent(0) == 2);
  REQUIRE(arr.extent(1) == 3);
  REQUIRE(arr.extent(2) == 4);
  REQUIRE(arr.extent(3) == 5);
  REQUIRE(arr.shape() == std::array<int, 4>{2, 3, 4, 5});
  REQUIRE(arr.size() == 2 * 3 * 4 * 5);

  auto view = arr.view();
  view(1, 2, 3, 4) = 1234;
  REQUIRE(arr(1, 2, 3, 4) == 1234);
  REQUIRE(&view(1, 2, 3, 4) == &arr(1, 2, 3, 4));
  REQUIRE(view(1, 2, 3, 4) == arr[(((1 * 3) + 2) * 4 + 3) * 5 + 4]);

  Array4D<int, backend::Host> other(2, 3, 4, 5);
  for (int i = 0; i < arr.size(); ++i) {
    other[i] = i;
  }
  arr.copy_from(other);

  for (int i = 0; i < arr.size(); ++i) {
    REQUIRE(arr[i] == i);
  }
}
