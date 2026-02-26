#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <limits>

#include "backend.hpp"
#include "cuda_compat.hpp"
#ifdef __CUDACC__
#include "cuda_util.cuh"
#endif  // __CUDACC__

namespace miso {

/// @brief 2D Array in general execution/memory space.
template <typename T, typename Backend = backend::Host> class Array2D;

/// @brief Lightweight non-owning view of 2D array data.
template <typename T> class Array2DView {
private:
  T *data_ = nullptr;
  int shape_[2] = {-1, -1};

  // Always constructed from Array2D
  // to ensure that the array size is within the numeric limits of int.
  template <typename, typename> friend class Array2D;
  Array2DView(T *data, int nx0, int nx1) noexcept
      : data_(data), shape_{nx0, nx1} {}

public:
  /// @brief Return pointer to the data.
  __host__ __device__ T *data() noexcept { return data_; }

  /// @brief Return const pointer to the data.
  __host__ __device__ const T *data() const noexcept { return data_; }

  /// @brief Return the array extent in each dimension.
  __host__ __device__ int extent(int dim) const noexcept {
    assert(dim >= 0 && dim < 2);
    return shape_[dim];
  }

  // The shape() method is not implemented in Array2DView
  // as returning shape from a __host__ __device__ interface is awkward.
  // Use extent() instead.

  /// @brief Return the total size of the array.
  __host__ __device__ int size() const noexcept { return shape_[0] * shape_[1]; }

  /// @brief Return reference to the element at the given indices.
  __host__ __device__ T &operator()(int i0, int i1) const noexcept {
    assert(data_);
    assert(i0 >= 0 && i0 < shape_[0]);
    assert(i1 >= 0 && i1 < shape_[1]);
    return data_[i0 * shape_[1] + i1];
  }

  /// @brief Return reference to the element at the given linear index.
  __host__ __device__ T &operator[](int idx) const noexcept {
    assert(data_);
    assert(idx >= 0 && idx < size());
    return data_[idx];
  }

  // Allow copy semantics
  __host__ __device__ Array2DView(const Array2DView &) noexcept = default;
  __host__ __device__ Array2DView &
  operator=(const Array2DView &) noexcept = default;
};

/// @brief 2D Array in host memory.
template <typename T> class Array2D<T, backend::Host> {
private:
  T *data_ = nullptr;
  std::array<int, 2> shape_ = {-1, -1};

public:
  Array2D(int nx0, int nx1) : shape_{nx0, nx1} {
    assert(shape_[0] > 0 && shape_[1] > 0);
    const std::size_t array_size =
        static_cast<size_t>(shape_[0]) * static_cast<size_t>(shape_[1]);
    assert(array_size <= static_cast<size_t>(std::numeric_limits<int>::max()));
    data_ = new T[array_size];
  }

  ~Array2D() { delete[] data_; }

  /// @brief Return a lightweight view of the array.
  Array2DView<T> view() noexcept {
    assert(data_);
    assert(shape_[0] > 0 && shape_[1] > 0);
    return Array2DView<T>(data_, shape_[0], shape_[1]);
  }

  /// @brief Return a constant lightweight view of the array.
  Array2DView<const T> view() const noexcept {
    assert(data_);
    assert(shape_[0] > 0 && shape_[1] > 0);
    return Array2DView<const T>(data_, shape_[0], shape_[1]);
  }

  /// @brief Return a constant lightweight view of the array.
  Array2DView<const T> const_view() const noexcept { return view(); }

  /// @brief Return pointer to the data.
  T *data() noexcept { return data_; }

  /// @brief Return const pointer to the data.
  const T *data() const noexcept { return data_; }

  /// @brief Return the array extent in each dimension.
  __host__ __device__ int extent(int dim) const noexcept {
    assert(dim >= 0 && dim < 2);
    return shape_[dim];
  }

  /// @brief Return the shape of the array in the given axis.
  std::array<int, 2> shape() const noexcept { return shape_; }

  /// @brief Return the total size of the array.
  int size() const noexcept { return shape_[0] * shape_[1]; }

  /// @brief Return reference to the element at the given indices.
  T &operator()(int i0, int i1) const noexcept {
    assert(data_);
    assert(i0 >= 0 && i0 < shape_[0]);
    assert(i1 >= 0 && i1 < shape_[1]);
    return data_[i0 * shape_[1] + i1];
  }

  /// @brief Return reference to the element at the given linear index.
  T &operator[](int idx) const noexcept {
    assert(data_);
    assert(idx >= 0 && idx < size());
    return data_[idx];
  }

  /// @brief Copy data from another Array2D in host memory.
  void copy_from(const Array2D<T, backend::Host> &other) {
    assert(other.data() && data());
    assert(other.shape() == shape());
    std::copy(other.data(), other.data() + other.size(), data());
  }

#ifdef __CUDACC__
  /// @brief Copy data from another Array2D in CUDA memory.
  void copy_from(const Array2D<T, backend::CUDA> &other) {
    assert(other.data() && data());
    assert(other.shape() == shape());
    MISO_CUDA_CHECK(cudaMemcpy(data(), other.data(), sizeof(T) * other.size(),
                               cudaMemcpyDeviceToHost));
  }

  /// @brief Copy data from another Array2D in CUDA memory (async).
  /// @details The caller is responsible for synchronizing the stream.
  void copy_from(const Array2D<T, backend::CUDA> &other, cudaStream_t stream) {
    assert(other.data() && data());
    assert(other.shape() == shape());
    MISO_CUDA_CHECK(cudaMemcpyAsync(data(), other.data(),
                                    sizeof(T) * other.size(),
                                    cudaMemcpyDeviceToHost, stream));
  }
#endif  // __CUDACC__

  // Prohibit copy semantics
  Array2D(const Array2D &) = delete;
  Array2D &operator=(const Array2D &) = delete;

  // Allow move semantics (allow uninitialized state)
  Array2D(Array2D &&other) noexcept : data_(other.data_), shape_(other.shape_) {
    other.data_ = nullptr;
    other.shape_ = std::array<int, 2>{-1, -1};
  }
  Array2D &operator=(Array2D &&other) noexcept {
    if (this != &other) {
      delete[] data_;
      data_ = other.data_;
      shape_ = other.shape_;
      other.data_ = nullptr;
      other.shape_ = std::array<int, 2>{-1, -1};
    }
    return *this;
  }
};

#ifdef __CUDACC__
/// @brief 2D Array in CUDA device memory.
template <typename T> class Array2D<T, backend::CUDA> {
private:
  T *data_ = nullptr;
  std::array<int, 2> shape_ = {-1, -1};

public:
  Array2D(int nx0, int nx1) : shape_{nx0, nx1} {
    assert(shape_[0] > 0 && shape_[1] > 0);
    const std::size_t array_size =
        static_cast<size_t>(shape_[0]) * static_cast<size_t>(shape_[1]);
    assert(array_size <= static_cast<size_t>(std::numeric_limits<int>::max()));
    MISO_CUDA_CHECK(cudaMalloc(&data_, sizeof(T) * array_size));
  }

  ~Array2D() {
    if (data_)
      cudaFree(data_);
    data_ = nullptr;
  }

  /// @brief Return a lightweight view of the array.
  Array2DView<T> view() noexcept {
    assert(data_);
    assert(shape_[0] > 0 && shape_[1] > 0);
    return Array2DView<T>(data_, shape_[0], shape_[1]);
  }

  /// @brief Return a constant lightweight view of the array.
  Array2DView<const T> view() const noexcept {
    assert(data_);
    assert(shape_[0] > 0 && shape_[1] > 0);
    return Array2DView<const T>(data_, shape_[0], shape_[1]);
  }

  /// @brief Return a constant lightweight view of the array.
  Array2DView<const T> const_view() const noexcept { return view(); }

  /// @brief Return pointer to the data.
  T *data() noexcept { return data_; }

  /// @brief Return const pointer to the data.
  const T *data() const noexcept { return data_; }

  /// @brief Return the array extent in each dimension.
  __host__ __device__ int extent(int dim) const noexcept {
    assert(dim >= 0 && dim < 2);
    return shape_[dim];
  }

  /// @brief Return the shape of the array in the given axis.
  std::array<int, 2> shape() const noexcept { return shape_; }

  /// @brief Return the total size of the array.
  int size() const noexcept { return shape_[0] * shape_[1]; }

  /// @brief Copy data from another Array2D in host memory.
  void copy_from(const Array2D<T, backend::Host> &other) {
    assert(other.data() && data());
    assert(other.shape() == shape());
    MISO_CUDA_CHECK(cudaMemcpy(data(), other.data(), sizeof(T) * size(),
                               cudaMemcpyHostToDevice));
  }

  /// @brief Copy data from another Array2D in CUDA memory.
  void copy_from(const Array2D<T, backend::CUDA> &other) {
    assert(other.data() && data());
    assert(other.shape() == shape());
    MISO_CUDA_CHECK(cudaMemcpy(data(), other.data(), sizeof(T) * size(),
                               cudaMemcpyDeviceToDevice));
  }

  /// @brief Copy data from another Array2D in host memory (async).
  /// @details The caller is responsible for synchronizing the stream.
  void copy_from(const Array2D<T, backend::Host> &other, cudaStream_t stream) {
    assert(other.data() && data());
    assert(other.shape() == shape());
    MISO_CUDA_CHECK(cudaMemcpyAsync(data(), other.data(), sizeof(T) * size(),
                                    cudaMemcpyHostToDevice, stream));
  }

  /// @brief Copy data from another Array2D in CUDA memory (async).
  /// @details The caller is responsible for synchronizing the stream.
  void copy_from(const Array2D<T, backend::CUDA> &other, cudaStream_t stream) {
    assert(other.data() && data());
    assert(other.shape() == shape());
    MISO_CUDA_CHECK(cudaMemcpyAsync(data(), other.data(), sizeof(T) * size(),
                                    cudaMemcpyDeviceToDevice, stream));
  }

  // Prohibit copy semantics
  Array2D(const Array2D &) = delete;
  Array2D &operator=(const Array2D &) = delete;

  // Allow move semantics (allow uninitialized state)
  Array2D(Array2D &&other) noexcept : data_(other.data_), shape_(other.shape_) {
    other.data_ = nullptr;
    other.shape_ = std::array<int, 2>{-1, -1};
  }
  Array2D &operator=(Array2D &&other) noexcept {
    if (this != &other) {
      if (data_)
        cudaFree(data_);
      data_ = other.data_;
      shape_ = other.shape_;
      other.data_ = nullptr;
      other.shape_ = std::array<int, 2>{-1, -1};
    }
    return *this;
  }
};
#endif  // __CUDACC__

}  // namespace miso
