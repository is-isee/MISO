#pragma once
#include <miso/backend.hpp>
#include <miso/cuda_compat.hpp>
#ifdef __CUDACC__
#include <miso/cuda_util.cuh>
#endif  // __CUDACC__

#ifdef __CUDACC__
#define MISO_LAMBDA [=] __host__ __device__
#else
#define MISO_LAMBDA [=]
#endif  // __CUDACC__

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

/// @brief 1D parallel (concurrent) for loop in the Host backend.
template <class F> inline void for_each(backend::Host, Range1D range, F f) {
  for (int i = range.begin; i < range.end; ++i) {
    f(i);
  }
}

/// @brief 3D parallel (concurrent) for loop in the Host backend.
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

/// @brief 1D parallel (concurrent) for loop in the CUDA backend.
template <class F> inline void for_each(backend::CUDA, Range1D range, F f) {
  int n = range.size();
  if (n == 0)
    return;  // Avoid launching kernel with zero threads
  int block = 256;
  int grid = (n + block - 1) / block;
  for_each_kernel<<<grid, block>>>(range, f);
  MISO_CUDA_CHECK(cudaGetLastError());
}

template <class F> __global__ void for_each_kernel(Range3D range, F f) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < range.size()) {
    Index3D p = range.unravel(idx);
    f(p.i0, p.i1, p.i2);
  }
}

/// @brief 3D parallel (concurrent) for loop in the CUDA backend.
template <class F> inline void for_each(backend::CUDA, Range3D range, F f) {
  int n = range.size();
  if (n == 0)
    return;  // Avoid launching kernel with zero threads
  int block = 256;
  int grid = (n + block - 1) / block;
  for_each_kernel<<<grid, block>>>(range, f);
  MISO_CUDA_CHECK(cudaGetLastError());
}

#endif

/// @brief 1D parallel reduction in the Host backend.
template <class T, class F, class Op>
inline T reduce(backend::Host, Range1D range, T init, F map, Op op) {
  T acc = init;
  for (int i = range.begin; i < range.end; ++i)
    acc = op(acc, map(i));
  return acc;
}

/// @brief 3D parallel reduction in the Host backend.
template <class T, class F, class Op>
inline T reduce(backend::Host, Range3D range, T init, F map, Op op) {
  T acc = init;
  for (int i0 = range.range0.begin; i0 < range.range0.end; ++i0)
    for (int i1 = range.range1.begin; i1 < range.range1.end; ++i1)
      for (int i2 = range.range2.begin; i2 < range.range2.end; ++i2)
        acc = op(acc, map(i0, i1, i2));
  return acc;
}

#ifdef __CUDACC__

template <typename T> struct ReduceHelper {
  T *buf0 = nullptr;
  T *buf1 = nullptr;
  int capacity = 0;  // number of T
  int block = 256;   // threads per block (1D)
  size_t shmem = 0;

  ReduceHelper() = default;
  explicit ReduceHelper(int block_) : block(block_) {}

  ~ReduceHelper() { reset(); }

  void reset() {
    if (buf0)
      MISO_CUDA_CHECK(cudaFree(buf0));
    if (buf1)
      MISO_CUDA_CHECK(cudaFree(buf1));
    buf0 = buf1 = nullptr;
    capacity = 0;
    shmem = 0;
  }

  void reserve(int n) {
    if (n <= capacity)
      return;
    reset();
    capacity = n;
    MISO_CUDA_CHECK(cudaMalloc(&buf0, sizeof(T) * capacity));
    MISO_CUDA_CHECK(cudaMalloc(&buf1, sizeof(T) * capacity));
    shmem = sizeof(T) * block;
  }
};

template <class T, class Op>
__device__ inline T block_reduce_shared(T *s, int tid, int n, Op op) {
  __syncthreads();
  for (int offset = (n + 1) >> 1; offset > 0; offset = (offset + 1) >> 1) {
    if (tid < offset && tid + offset < n) {
      s[tid] = op(s[tid], s[tid + offset]);
    }
    __syncthreads();
    if (offset == 1)
      break;
  }
  return s[0];
}

template <class T, class Range, class F, class Op>
__global__ void reduce_stage_kernel(Range range, T init, F map, Op op,
                                    T *__restrict__ out) {
  extern __shared__ T s[];
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid;

  T v = init;
  if (idx < range.size()) {
    if constexpr (std::is_same_v<Range, Range1D>) {
      const int i = range.unravel(idx);
      v = map(i);
    } else {
      Index3D p = range.unravel(idx);
      v = map(p.i0, p.i1, p.i2);
    }
  }
  s[tid] = v;
  block_reduce_shared(s, tid, blockDim.x, op);
  if (tid == 0)
    out[blockIdx.x] = s[0];
}

template <class T, class Op>
__global__ void reduce_stage_array_kernel(const T *__restrict__ in, int n, T init,
                                          Op op, T *__restrict__ out) {
  extern __shared__ T s[];
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid;
  s[tid] = (idx < n) ? in[idx] : init;
  block_reduce_shared(s, tid, blockDim.x, op);
  if (tid == 0)
    out[blockIdx.x] = s[0];
}

template <class T, class Range, class F, class Op>
inline T reduce(backend::CUDA, Range range, T init, F map, Op op,
                ReduceHelper<T> &helper, cudaStream_t stream = 0) {
  const int n = range.size();
  if (n <= 0)
    return init;

  const int block = helper.block;
  int grid = (n + block - 1) / block;

  // workspace needs at least max(grid, ...) but easiest: reserve(n) is enough
  helper.reserve(n);

  // stage 1: map+reduce into buf0[grid]
  reduce_stage_kernel<T, Range, F, Op>
      <<<grid, block, helper.shmem, stream>>>(range, init, map, op, helper.buf0);
  MISO_CUDA_CHECK(cudaGetLastError());

  // stage 2+: keep reducing buf0 -> buf1 until 1 element
  int cur_n = grid;
  T *in = helper.buf0;
  T *out = helper.buf1;

  while (cur_n > 1) {
    int g = (cur_n + block - 1) / block;
    reduce_stage_array_kernel<T, Op>
        <<<g, block, helper.shmem, stream>>>(in, cur_n, init, op, out);
    MISO_CUDA_CHECK(cudaGetLastError());
    cur_n = g;
    T *tmp = in;
    in = out;
    out = tmp;
  }

  // copy back one value
  T h;
  MISO_CUDA_CHECK(
      cudaMemcpyAsync(&h, in, sizeof(T), cudaMemcpyDeviceToHost, stream));
  MISO_CUDA_CHECK(cudaStreamSynchronize(stream));
  return h;
}

/// @brief 1D parallel reduction in the CUDA backend.
template <class T, class F, class Op>
inline T reduce(backend::CUDA b, Range1D range, T init, F map, Op op,
                ReduceHelper<T> &helper, cudaStream_t stream = 0) {
  return reduce<T, Range1D, F, Op>(b, range, init, map, op, helper, stream);
}

/// @brief 3D parallel reduction in the CUDA backend.
template <class T, class F, class Op>
inline T reduce(backend::CUDA b, Range3D range, T init, F map, Op op,
                ReduceHelper<T> &helper, cudaStream_t stream = 0) {
  return reduce<T, Range3D, F, Op>(b, range, init, map, op, helper, stream);
}

#endif  // __CUDACC__

}  // namespace miso
