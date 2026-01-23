#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <miso/array1d.hpp>
#include <miso/array3d.hpp>
#include <miso/execution.hpp>

using namespace miso;

TEST_CASE("Test for_each 1D CPU" * doctest::test_suite("execution")) {
  Range1D range{2, 4};
  Array1D<int, backend::Host> arr(5);

  auto view = arr.view();
  for_each<backend::Host>(range, [&](int i) { view[i] = i * i; });

  // This also works, but users should use the view interface as API.
  // for_each<backend::Host>(range, [&](int i) { arr[i] = i * i; });

  for (int i = 2; i < 4; ++i)
    REQUIRE(view[i] == i * i);
}

TEST_CASE("Test for_each 3D CPU" * doctest::test_suite("execution")) {
  Range3D range{{1, 2}, {1, 3}, {0, 4}};
  Array3D<int, backend::Host> arr(2, 3, 4);

  auto view = arr.view();
  for_each<backend::Host>(
      range, MISO_LAMBDA(int i, int j, int k) { view(i, j, k) = i + j + k; });

  for (int i = 1; i < 2; ++i) {
    for (int j = 1; j < 3; ++j) {
      for (int k = 0; k < 4; ++k) {
        REQUIRE(view(i, j, k) == i + j + k);
      }
    }
  }
}
