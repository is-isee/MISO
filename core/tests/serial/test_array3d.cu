#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <cassert>
#include <doctest/doctest.h>

#define USE_CUDA
#include <miso/array3d.hpp>

using namespace miso;

__host__ __device__ inline double ref_value(int i, int j, int k) {
  return static_cast<double>(i * 100 + j * 10 + k);
}

__global__ void test_array3d_kernel(Array3DView<double> arr) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < arr.extent(0) && j < arr.extent(1) && k < arr.extent(2)) {
    arr(i, j, k) = ref_value(i, j, k);
  }
}

TEST_CASE("Test Array3D GPU" * doctest::test_suite("array3d")) {
  Array3D<double, HostSpace> arr(3, 4, 5);
  Array3D<double, CUDASpace> arr_d(3, 4, 5);

  const auto [nx0, nx1, nx2] = arr.shape();
  for (int i = 0; i < nx0; ++i) {
    for (int j = 0; j < nx1; ++j) {
      for (int k = 0; k < nx2; ++k) {
        arr(i, j, k) = ref_value(i, j, k);
      }
    }
  }

  arr_d.copy_from(arr);
  test_array3d_kernel<<<dim3(1, 1, 1), dim3(3, 4, 5)>>>(arr_d.view());

  arr.copy_from(arr_d);
  for (int i = 0; i < nx0; ++i) {
    for (int j = 0; j < nx1; ++j) {
      for (int k = 0; k < nx2; ++k) {
        REQUIRE(std::abs(arr(i, j, k) - ref_value(i, j, k)) < 1e-9);
      }
    }
  }
}
