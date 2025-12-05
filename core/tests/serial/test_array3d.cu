#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "array3d_cpu.hpp"
#include "array3d_gpu.cuh"
#include <cassert>
#include <doctest/doctest.h>

__global__ void test_array3d_kernel(double *data, int size_x, int size_y,
                                    int size_z) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x < size_x && y < size_y && z < size_z) {
    data[z * size_x * size_y + y * size_x + x] = 1.0;  // Set all elements to 1.0
  }
}

TEST_CASE("Test Array3D GPU" * doctest::test_suite("array3d")) {
  Array3D<double> arr(3, 4, 5);
  Array3DDevice<double> arr_d(3, 4, 5);

  // Check access
  arr(1, 2, 3) = 42.0;
  REQUIRE(arr(1, 2, 3) == 42.0);

  arr_d.copy_from_host(arr);

  test_array3d_kernel<<<dim3(1, 1, 1), dim3(3, 4, 5)>>>(
      arr_d.data(), arr.size_x(), arr.size_y(), arr.size_z());
  arr_d.copy_to_host(arr);
  REQUIRE(arr(1, 2, 3) == 1.0);
}
