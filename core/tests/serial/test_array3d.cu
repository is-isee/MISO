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
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x < arr.size_x() && y < arr.size_y() && z < arr.size_z()) {
    arr(x, y, z) = ref_value(x, y, z);
  }
}

TEST_CASE("Test Array3D GPU" * doctest::test_suite("array3d")) {
  Array3D<double, HostSpace> arr(3, 4, 5);
  Array3D<double, CUDASpace> arr_d(3, 4, 5);

  for (int i = 0; i < arr.size_x(); ++i) {
    for (int j = 0; j < arr.size_y(); ++j) {
      for (int k = 0; k < arr.size_z(); ++k) {
        arr(i, j, k) = ref_value(i, j, k);
      }
    }
  }

  arr_d.copy_from(arr);
  test_array3d_kernel<<<dim3(1, 1, 1), dim3(3, 4, 5)>>>(arr_d.view());

  arr.copy_from(arr_d);
  for (int i = 0; i < arr.size_x(); ++i) {
    for (int j = 0; j < arr.size_y(); ++j) {
      for (int k = 0; k < arr.size_z(); ++k) {
        REQUIRE(std::abs(arr(i, j, k) - ref_value(i, j, k)) < 1e-9);
      }
    }
  }
}
