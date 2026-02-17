#pragma once

#include <algorithm>
#include <cassert>
#include <limits>

#include "backend.hpp"
#include "cuda_compat.hpp"
#ifdef __CUDACC__
#include "cuda_util.cuh"
#endif  // __CUDACC__

namespace miso {

/// @brief 1D Array in general execution/memory space.
template <typename T, typename Backend = backend::Host> class Array1D;

/// @brief Lightweight non-owning view of 1D array data.
template <typename T> class Array1DView {
private:
  T *data_ = nullptr;
  int size_ = -1;

  // Always constructed from Array1D
  // to ensure that the array size is within the numeric limits of int.
  template <typename, typename> friend class Array1D;
  Array1DView(T *data, int size) noexcept : data_(data), size_(size) {}

public:
  /// @brief Return pointer to the data.
  __host__ __device__ T *data() noexcept { return data_; }

  /// @brief Return const pointer to the data.
  __host__ __device__ const T *data() const noexcept { return data_; }

  /// @brief Return the size of the array.
  __host__ __device__ int size() const noexcept { return size_; }

  /// @brief Return reference to the element at the given linear index.
  __host__ __device__ T &operator[](int idx) const noexcept {
    assert(data_);
    assert(idx >= 0 && idx < size());
    return data_[idx];
  }

  // Allow copy semantics
  __host__ __device__ Array1DView(const Array1DView &) noexcept = default;
  __host__ __device__ Array1DView &
  operator=(const Array1DView &) noexcept = default;
};

/// @brief 1D Array in host memory.
template <typename T> class Array1D<T, backend::Host> {
private:
  T *data_ = nullptr;
  int size_ = -1;

public:
  Array1D(int nx0) : size_(nx0) {
    assert(size_ > 0);
    data_ = new T[size_];
  }

  ~Array1D() { delete[] data_; }

  /// @brief Return a lightweight view of the array.
  Array1DView<T> view() noexcept {
    assert(data_ && size_ > 0);
    return Array1DView<T>(data_, size_);
  }

  /// @brief Return a constant lightweight view of the array.
  Array1DView<const T> view() const noexcept {
    assert(data_ && size_ > 0);
    return Array1DView<const T>(data_, size_);
  }

  /// @brief Return a constant lightweight view of the array.
  Array1DView<const T> const_view() const noexcept { return view(); }

  /// @brief Return pointer to the data.
  T *data() noexcept { return data_; }

  /// @brief Return const pointer to the data.
  const T *data() const noexcept { return data_; }

  /// @brief Return the size of the array.
  int size() const noexcept { return size_; }

  /// @brief Return reference to the element at the given linear index.
  T &operator[](int idx) const noexcept {
    assert(data_);
    assert(idx >= 0 && idx < size());
    return data_[idx];
  }

  /// @brief Copy data from another Array1D in host memory.
  void copy_from(const Array1D<T, backend::Host> &other) {
    assert(other.data() && data());
    assert(other.size() == size());
    std::copy(other.data(), other.data() + other.size(), data());
  }

#ifdef __CUDACC__
  /// @brief Copy data from another Array1D in CUDA memory.
  void copy_from(const Array1D<T, backend::CUDA> &other) {
    assert(other.data() && data());
    assert(other.size() == size());
    MISO_CUDA_CHECK(cudaMemcpy(data(), other.data(), sizeof(T) * other.size(),
                               cudaMemcpyDeviceToHost));
  }

  /// @brief Copy data from another Array1D in CUDA memory (async).
  /// @details The caller is responsible for synchronizing the stream.
  void copy_from(const Array1D<T, backend::CUDA> &other, cudaStream_t stream) {
    assert(other.data() && data());
    assert(other.size() == size());
    MISO_CUDA_CHECK(cudaMemcpyAsync(data(), other.data(),
                                    sizeof(T) * other.size(),
                                    cudaMemcpyDeviceToHost, stream));
  }
#endif  // __CUDACC__

  // Prohibit copy semantics
  Array1D(const Array1D &) = delete;
  Array1D &operator=(const Array1D &) = delete;

  // Allow move semantics (allow uninitialized state)
  Array1D(Array1D &&other) noexcept : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = -1;
  }
  Array1D &operator=(Array1D &&other) noexcept {
    if (this != &other) {
      delete[] data_;
      data_ = other.data_;
      size_ = other.size_;
      other.data_ = nullptr;
      other.size_ = -1;
    }
    return *this;
  }
};

#ifdef __CUDACC__
/// @brief 1D Array in CUDA device memory.
template <typename T> class Array1D<T, backend::CUDA> {
private:
  T *data_ = nullptr;
  int size_ = -1;

public:
  Array1D(int size) : size_(size) {
    assert(size_ > 0);
    MISO_CUDA_CHECK(cudaMalloc(&data_, sizeof(T) * size_));
  }

  ~Array1D() {
    if (data_)
      cudaFree(data_);
    data_ = nullptr;
  }

  /// @brief Return a lightweight view of the array.
  Array1DView<T> view() noexcept {
    assert(data_ && size_ > 0);
    return Array1DView<T>(data_, size_);
  }

  /// @brief Return a constant lightweight view of the array.
  Array1DView<const T> view() const noexcept {
    assert(data_ && size_ > 0);
    return Array1DView<const T>(data_, size_);
  }

  /// @brief Return a constant lightweight view of the array.
  Array1DView<const T> const_view() const noexcept { return view(); }

  /// @brief Return pointer to the data.
  T *data() noexcept { return data_; }

  /// @brief Return const pointer to the data.
  const T *data() const noexcept { return data_; }

  /// @brief Return the size of the array.
  int size() const noexcept { return size_; }

  /// @brief Copy data from another Array1D in host memory.
  void copy_from(const Array1D<T, backend::Host> &other) {
    assert(other.data() && data());
    assert(other.size() == size());
    MISO_CUDA_CHECK(cudaMemcpy(data(), other.data(), sizeof(T) * size(),
                               cudaMemcpyHostToDevice));
  }

  /// @brief Copy data from another Array1D in CUDA memory.
  void copy_from(const Array1D<T, backend::CUDA> &other) {
    assert(other.data() && data());
    assert(other.size() == size());
    MISO_CUDA_CHECK(cudaMemcpy(data(), other.data(), sizeof(T) * size(),
                               cudaMemcpyDeviceToDevice));
  }

  /// @brief Copy data from another Array1D in host memory (async).
  /// @details The caller is responsible for synchronizing the stream.
  void copy_from(const Array1D<T, backend::Host> &other, cudaStream_t stream) {
    assert(other.data() && data());
    assert(other.size() == size());
    MISO_CUDA_CHECK(cudaMemcpyAsync(data(), other.data(), sizeof(T) * size(),
                                    cudaMemcpyHostToDevice, stream));
  }

  /// @brief Copy data from another Array1D in CUDA memory (async).
  /// @details The caller is responsible for synchronizing the stream.
  void copy_from(const Array1D<T, backend::CUDA> &other, cudaStream_t stream) {
    assert(other.data() && data());
    assert(other.size() == size());
    MISO_CUDA_CHECK(cudaMemcpyAsync(data(), other.data(), sizeof(T) * size(),
                                    cudaMemcpyDeviceToDevice, stream));
  }

  // Prohibit copy semantics
  Array1D(const Array1D &) = delete;
  Array1D &operator=(const Array1D &) = delete;

  // Allow move semantics (allow uninitialized state)
  Array1D(Array1D &&other) noexcept : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = -1;
  }
  Array1D &operator=(Array1D &&other) noexcept {
    if (this != &other) {
      if (data_)
        cudaFree(data_);
      data_ = other.data_;
      size_ = other.size_;
      other.data_ = nullptr;
      other.size_ = -1;
    }
    return *this;
  }
};
#endif  // __CUDACC__

}  // namespace miso
