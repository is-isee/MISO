#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <cassert>
#include <doctest/doctest.h>

#include <miso/array2d.hpp>

using namespace miso;

__host__ __device__ inline double ref_value(int i, int j) {
  return static_cast<double>(i * 10 + j);
}

__global__ void test_array2d_kernel(Array2DView<double> arr) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < arr.extent(0) && j < arr.extent(1)) {
    arr(i, j) = ref_value(i, j);
  }
}

TEST_CASE("Test Array2D GPU" * doctest::test_suite("array2d")) {
  Array2D<double, backend::Host> arr(3, 4);
  Array2D<double, backend::CUDA> arr_d(3, 4);

  const auto [nx0, nx1] = arr.shape();
  for (int i = 0; i < nx0; ++i) {
    for (int j = 0; j < nx1; ++j) {
      arr(i, j) = ref_value(i, j);
    }
  }

  arr_d.copy_from(arr);
  test_array2d_kernel<<<dim3(1, 1, 1), dim3(3, 4, 1)>>>(arr_d.view());

  arr.copy_from(arr_d);
  for (int i = 0; i < nx0; ++i) {
    for (int j = 0; j < nx1; ++j) {
      REQUIRE(std::abs(arr(i, j) - ref_value(i, j)) < 1e-9);
    }
  }
}
