#pragma once
#include <miso/backend.hpp>
#include <miso/cuda_compat.hpp>

#define MISO_LAMBDA [=] __host__ __device__

namespace miso {

/// @brief 1D range structure.
struct Range1D {
  int begin, end;

  /// @brief Get the size of the range.
  /// @details The size is floored at zero.
  __device__ __host__ int size() const { return end > begin ? (end - begin) : 0; }

  /// @brief Convert a linear index to a 1D index within the range.
  __device__ __host__ int unravel(int i) const { return begin + i; }
};

/// @brief 3D index structure.
struct Index3D {
  int i0, i1, i2;
};

/// @brief 3D range structure.
struct Range3D {
  Range1D range0, range1, range2;

  /// @brief Get the total size of the 3D range.
  /// @details The total size is assumed to fit in an int.
  __device__ __host__ int size() const {
    return range0.size() * range1.size() * range2.size();
  }

  /// @brief Convert a linear index to a 3D index within the range.
  __device__ __host__ Index3D unravel(int idx) const {
    int n1 = range1.size();
    int n2 = range2.size();
    int i2 = range2.begin + (idx % n2);
    idx /= n2;
    int i1 = range1.begin + (idx % n1);
    int i0 = range0.begin + (idx / n1);
    return {i0, i1, i2};
  }
};

template <class F> inline void for_each(backend::Host, Range1D range, F f) {
  for (int i = range.begin; i < range.end; ++i) {
    f(i);
  }
}

template <class F> inline void for_each(backend::Host, Range3D range, F f) {
  for (int i0 = range.range0.begin; i0 < range.range0.end; ++i0) {
    for (int i1 = range.range1.begin; i1 < range.range1.end; ++i1) {
      for (int i2 = range.range2.begin; i2 < range.range2.end; ++i2) {
        f(i0, i1, i2);
      }
    }
  }
}

#ifdef __CUDACC__

template <class F> __global__ void for_each_kernel(Range1D range, F f) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < range.size()) {
    int i = range.unravel(idx);
    f(i);
  }
}

template <class F> inline void for_each(backend::CUDA, Range1D range, F f) {
  int n = range.size();
  if (n == 0)
    return;  // Avoid launching kernel with zero threads
  int block = 256;
  int grid = (n + block - 1) / block;
  for_each_kernel<<<grid, block>>>(range, f);
}

template <class F> __global__ void for_each_kernel(Range3D range, F f) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < range.size()) {
    Index3D p = range.unravel(idx);
    f(p.i0, p.i1, p.i2);
  }
}

template <class F> inline void for_each(backend::CUDA, Range3D range, F f) {
  int n = range.size();
  if (n == 0)
    return;  // Avoid launching kernel with zero threads
  int block = 256;
  int grid = (n + block - 1) / block;
  for_each_kernel<<<grid, block>>>(range, f);
}

#endif

}  // namespace miso
