#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <cassert>
#include <doctest/doctest.h>

#include <miso/array1d.hpp>

using namespace miso;

__host__ __device__ inline double ref_value(int i) {
  return static_cast<double>(i * i);
}

__global__ void test_kernel(Array1DView<double> arr) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < arr.size()) {
    arr[i] = ref_value(i);
  }
}

TEST_CASE("Test Array1D GPU" * doctest::test_suite("array1d")) {
  Array1D<double, backend::Host> arr(3);
  Array1D<double, backend::CUDA> arr_d(3);

  for (int i = 0; i < arr.size(); ++i) {
    arr[i] = ref_value(i);
  }

  arr_d.copy_from(arr);
  test_kernel<<<1, 3>>>(arr_d.view());

  arr.copy_from(arr_d);
  for (int i = 0; i < arr.size(); ++i) {
    REQUIRE(std::abs(arr[i] - ref_value(i)) < 1e-9);
  }
}
