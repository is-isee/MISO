#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <miso/array1d.hpp>
#include <miso/array3d.hpp>
#include <miso/execution.hpp>

using namespace miso;

TEST_CASE("Test for_each 1D CUDA" * doctest::test_suite("execution")) {
  Range1D range{2, 5};
  Array1D<int, backend::CUDA> arr(5);

  auto view = arr.view();
  for_each(backend::CUDA{}, range, MISO_LAMBDA(int i) { view[i] = i * i; });

  Array1D<int, backend::Host> arr_host(5);
  arr_host.copy_from(arr);
  const auto view_host = arr_host.view();
  for (int i = 2; i < 5; ++i) {
    REQUIRE(view_host[i] == i * i);
  }
}

TEST_CASE("Test for_each 3D CUDA" * doctest::test_suite("execution")) {
  Range3D range{{1, 2}, {1, 3}, {0, 4}};
  Array3D<int, backend::CUDA> arr(2, 3, 4);

  auto view = arr.view();
  for_each(
      backend::CUDA{}, range,
      MISO_LAMBDA(int i, int j, int k) { view(i, j, k) = i + j + k; });

  Array3D<int, backend::Host> arr_host(2, 3, 4);
  arr_host.copy_from(arr);
  const auto view_host = arr_host.view();
  for (int i = 1; i < 2; ++i) {
    for (int j = 1; j < 3; ++j) {
      for (int k = 0; k < 4; ++k) {
        REQUIRE(view_host(i, j, k) == i + j + k);
      }
    }
  }
}
