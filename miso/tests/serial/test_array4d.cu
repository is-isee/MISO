#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <miso/array4d.hpp>

using namespace miso;

__host__ __device__ inline double ref_value4d(int i, int j, int k, int l) {
  return static_cast<double>(i * 1000 + j * 100 + k * 10 + l);
}

__global__ void test_array4d_kernel(Array4DView<double> arr) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < arr.extent(0) && j < arr.extent(1) && k < arr.extent(2)) {
    for (int l = 0; l < arr.extent(3); ++l) {
      arr(i, j, k, l) = ref_value4d(i, j, k, l);
    }
  }
}

TEST_CASE("Test Array4D GPU" * doctest::test_suite("array4d")) {
  Array4D<double, backend::Host> arr(2, 3, 4, 5);
  Array4D<double, backend::CUDA> arr_d(2, 3, 4, 5);

  const auto [nx0, nx1, nx2, nx3] = arr.shape();
  for (int i = 0; i < nx0; ++i) {
    for (int j = 0; j < nx1; ++j) {
      for (int k = 0; k < nx2; ++k) {
        for (int l = 0; l < nx3; ++l) {
          arr(i, j, k, l) = ref_value4d(i, j, k, l);
        }
      }
    }
  }

  arr_d.copy_from(arr);
  test_array4d_kernel<<<dim3(1, 1, 1), dim3(2, 3, 4)>>>(arr_d.view());

  arr.copy_from(arr_d);
  for (int i = 0; i < nx0; ++i) {
    for (int j = 0; j < nx1; ++j) {
      for (int k = 0; k < nx2; ++k) {
        for (int l = 0; l < nx3; ++l) {
          REQUIRE(std::abs(arr(i, j, k, l) - ref_value4d(i, j, k, l)) < 1e-9);
        }
      }
    }
  }
}
