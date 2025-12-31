#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <limits>

#include <miso/cuda_compat.hpp>
#include <miso/policy.hpp>
#ifdef USE_CUDA
#include <miso/cuda_util.cuh>
#endif  // USE_CUDA

namespace miso {

/// @brief 3D Array in general execution/memory space.
template <typename T, typename Space = HostSpace> class Array3D;

/// @brief Lightweight non-owning view of MHD fields.
template <typename T> class Array3DView {
private:
  T *data_ = nullptr;
  int shape_[3] = {-1, -1, -1};

  // Always constructed from Array3D
  // to ensure that the array size is within the numeric limits of int.
  template <typename, typename> friend class Array3D;
  __host__ __device__ Array3DView(T *data, int nx0, int nx1, int nx2) noexcept
      : data_(data), shape_{nx0, nx1, nx2} {}

public:
  /// @brief Return pointer to the data.
  __host__ __device__ T *data() noexcept { return data_; }

  /// @brief Return const pointer to the data.
  __host__ __device__ const T *data() const noexcept { return data_; }

  /// @brief Return the array extent in each dimension.
  __host__ __device__ int extent(int dim) const noexcept {
    assert(dim >= 0 && dim < 3);
    return shape_[dim];
  }

  // The shape() method is not implemented in Array3DView
  // as returning shape from a __host__ __device__ interface is awkward.
  // Use extent() instead.

  /// @brief Return the total size of the array.
  __host__ __device__ int size() const noexcept {
    return shape_[0] * shape_[1] * shape_[2];
  }

  /// @brief Return reference to the element at the given indices.
  __host__ __device__ T &operator()(int i0, int i1, int i2) noexcept {
    assert(data_);
    assert(i0 >= 0 && i0 < shape_[0]);
    assert(i1 >= 0 && i1 < shape_[1]);
    assert(i2 >= 0 && i2 < shape_[2]);
    return data_[(i0 * shape_[1] + i1) * shape_[2] + i2];
  }

  /// @brief Return const reference to the element at the given indices.
  __host__ __device__ const T &operator()(int i0, int i1, int i2) const noexcept {
    assert(data_);
    assert(i0 >= 0 && i0 < shape_[0]);
    assert(i1 >= 0 && i1 < shape_[1]);
    assert(i2 >= 0 && i2 < shape_[2]);
    return data_[(i0 * shape_[1] + i1) * shape_[2] + i2];
  }

  /// @brief Return reference to the element at the given linear index.
  __host__ __device__ T &operator[](int idx) noexcept {
    assert(data_);
    assert(idx >= 0 && idx < size());
    return data_[idx];
  }

  /// @brief Return const reference to the element at the given linear index.
  __host__ __device__ const T &operator[](int idx) const noexcept {
    assert(data_);
    assert(idx >= 0 && idx < size());
    return data_[idx];
  }

  // Allow copy semantics
  __host__ __device__ Array3DView(const Array3DView &) noexcept = default;
  __host__ __device__ Array3DView &
  operator=(const Array3DView &) noexcept = default;
};

/// @brief 3D Array in host memory.
template <typename T> class Array3D<T, HostSpace> {
private:
  T *data_ = nullptr;
  std::array<int, 3> shape_ = {-1, -1, -1};

public:
  Array3D(int nx0, int nx1, int nx2) : shape_{nx0, nx1, nx2} {
    assert(shape_[0] > 0 && shape_[1] > 0 && shape_[2] > 0);
    const std::size_t array_size = static_cast<size_t>(shape_[0]) *
                                   static_cast<size_t>(shape_[1]) *
                                   static_cast<size_t>(shape_[2]);
    assert(array_size <= static_cast<size_t>(std::numeric_limits<int>::max()));
    data_ = new T[array_size];
  }

  ~Array3D() { delete[] data_; }

  /// @brief Return a lightweight view of the array.
  Array3DView<T> view() noexcept {
    assert(data_);
    assert(shape_[0] > 0 && shape_[1] > 0 && shape_[2] > 0);
    return Array3DView<T>(data_, shape_[0], shape_[1], shape_[2]);
  }

  /// @brief Return a constant lightweight view of the array.
  Array3DView<const T> view() const noexcept {
    assert(data_);
    assert(shape_[0] > 0 && shape_[1] > 0 && shape_[2] > 0);
    return Array3DView<const T>(data_, shape_[0], shape_[1], shape_[2]);
  }

  /// @brief Return pointer to the data.
  T *data() noexcept { return data_; }

  /// @brief Return const pointer to the data.
  const T *data() const noexcept { return data_; }

  /// @brief Return the array extent in each dimension.
  __host__ __device__ int extent(int dim) const noexcept {
    assert(dim >= 0 && dim < 3);
    return shape_[dim];
  }

  /// @brief Return the shape of the array in the given axis.
  std::array<int, 3> shape() const noexcept { return shape_; }

  /// @brief Return the total size of the array.
  int size() const noexcept { return shape_[0] * shape_[1] * shape_[2]; }

  /// @brief Return reference to the element at the given indices.
  T &operator()(int i0, int i1, int i2) noexcept {
    assert(data_);
    assert(i0 >= 0 && i0 < shape_[0]);
    assert(i1 >= 0 && i1 < shape_[1]);
    assert(i2 >= 0 && i2 < shape_[2]);
    return data_[(i0 * shape_[1] + i1) * shape_[2] + i2];
  }

  /// @brief Return const reference to the element at the given indices.
  const T &operator()(int i0, int i1, int i2) const noexcept {
    assert(data_);
    assert(i0 >= 0 && i0 < shape_[0]);
    assert(i1 >= 0 && i1 < shape_[1]);
    assert(i2 >= 0 && i2 < shape_[2]);
    return data_[(i0 * shape_[1] + i1) * shape_[2] + i2];
  }

  /// @brief Return reference to the element at the given linear index.
  T &operator[](int idx) noexcept {
    assert(data_);
    assert(idx >= 0 && idx < size());
    return data_[idx];
  }

  /// @brief Return const reference to the element at the given linear index.
  const T &operator[](int idx) const noexcept {
    assert(data_);
    assert(idx >= 0 && idx < size());
    return data_[idx];
  }

  /// @brief Copy data from another Array3D in host memory.
  void copy_from(const Array3D<T, HostSpace> &other) {
    assert(other.data() && data());
    assert(other.shape() == shape());
    std::copy(other.data(), other.data() + other.size(), data());
  }

#ifdef USE_CUDA
  /// @brief Copy data from another Array3D in CUDA memory.
  void copy_from(const Array3D<T, CUDASpace> &other) {
    assert(other.data() && data());
    assert(other.shape() == shape());
    MISO_CUDA_CHECK(cudaMemcpy(data(), other.data(), sizeof(T) * other.size(),
                               cudaMemcpyDeviceToHost));
  }

  /// @brief Copy data from another Array3D in CUDA memory (async).
  /// @details The caller is responsible for synchronizing the stream.
  void copy_from(const Array3D<T, CUDASpace> &other, cudaStream_t stream) {
    assert(other.data() && data());
    assert(other.shape() == shape());
    MISO_CUDA_CHECK(cudaMemcpyAsync(data(), other.data(),
                                    sizeof(T) * other.size(),
                                    cudaMemcpyDeviceToHost, stream));
  }
#endif  // USE_CUDA

  // Prohibit copy semantics
  Array3D(const Array3D &) = delete;
  Array3D &operator=(const Array3D &) = delete;

  // Allow move semantics (allow uninitialized state)
  Array3D(Array3D &&other) noexcept : data_(other.data_), shape_(other.shape_) {
    other.data_ = nullptr;
    other.shape_ = std::array<int, 3>{-1, -1, -1};
  }
  Array3D &operator=(Array3D &&other) noexcept {
    if (this != &other) {
      delete[] data_;
      data_ = other.data_;
      shape_ = other.shape_;
      other.data_ = nullptr;
      other.shape_ = std::array<int, 3>{-1, -1, -1};
    }
    return *this;
  }
};

#ifdef USE_CUDA
/// @brief 3D Array in CUDA device memory.
template <typename T> class Array3D<T, CUDASpace> {
private:
  T *data_ = nullptr;
  std::array<int, 3> shape_ = {-1, -1, -1};

public:
  Array3D(int nx0, int nx1, int nx2) : shape_{nx0, nx1, nx2} {
    assert(shape_[0] > 0 && shape_[1] > 0 && shape_[2] > 0);
    const std::size_t array_size = static_cast<size_t>(shape_[0]) *
                                   static_cast<size_t>(shape_[1]) *
                                   static_cast<size_t>(shape_[2]);
    assert(array_size <= static_cast<size_t>(std::numeric_limits<int>::max()));
    MISO_CUDA_CHECK(cudaMalloc(&data_, sizeof(T) * array_size));
  }

  ~Array3D() {
    if (data_)
      cudaFree(data_);
    data_ = nullptr;
  }

  /// @brief Return a lightweight view of the array.
  Array3DView<T> view() noexcept {
    assert(data_);
    assert(shape_[0] > 0 && shape_[1] > 0 && shape_[2] > 0);
    return Array3DView<T>(data_, shape_[0], shape_[1], shape_[2]);
  }

  /// @brief Return a constant lightweight view of the array.
  Array3DView<const T> view() const noexcept {
    assert(data_);
    assert(shape_[0] > 0 && shape_[1] > 0 && shape_[2] > 0);
    return Array3DView<const T>(data_, shape_[0], shape_[1], shape_[2]);
  }

  /// @brief Return pointer to the data.
  T *data() noexcept { return data_; }

  /// @brief Return const pointer to the data.
  const T *data() const noexcept { return data_; }

  /// @brief Return the array extent in each dimension.
  __host__ __device__ int extent(int dim) const noexcept {
    assert(dim >= 0 && dim < 3);
    return shape_[dim];
  }

  /// @brief Return the shape of the array in the given axis.
  std::array<int, 3> shape() const noexcept { return shape_; }

  /// @brief Return the total size of the array.
  int size() const noexcept { return shape_[0] * shape_[1] * shape_[2]; }

  /// @brief Copy data from another Array3D in host memory.
  void copy_from(const Array3D<T, HostSpace> &other) {
    assert(other.data() && data());
    assert(other.shape() == shape());
    MISO_CUDA_CHECK(cudaMemcpy(data(), other.data(), sizeof(T) * size(),
                               cudaMemcpyHostToDevice));
  }

  /// @brief Copy data from another Array3D in CUDA memory.
  void copy_from(const Array3D<T, CUDASpace> &other) {
    assert(other.data() && data());
    assert(other.shape() == shape());
    MISO_CUDA_CHECK(cudaMemcpy(data(), other.data(), sizeof(T) * size(),
                               cudaMemcpyDeviceToDevice));
  }

  /// @brief Copy data from another Array3D in host memory (async).
  /// @details The caller is responsible for synchronizing the stream.
  void copy_from(const Array3D<T, HostSpace> &other, cudaStream_t stream) {
    assert(other.data() && data());
    assert(other.shape() == shape());
    MISO_CUDA_CHECK(cudaMemcpyAsync(data(), other.data(), sizeof(T) * size(),
                                    cudaMemcpyHostToDevice, stream));
  }

  /// @brief Copy data from another Array3D in CUDA memory (async).
  /// @details The caller is responsible for synchronizing the stream.
  void copy_from(const Array3D<T, CUDASpace> &other, cudaStream_t stream) {
    assert(other.data() && data());
    assert(other.shape() == shape());
    MISO_CUDA_CHECK(cudaMemcpyAsync(data(), other.data(), sizeof(T) * size(),
                                    cudaMemcpyDeviceToDevice, stream));
  }

  // Prohibit copy semantics
  Array3D(const Array3D &) = delete;
  Array3D &operator=(const Array3D &) = delete;

  // Allow move semantics (allow uninitialized state)
  Array3D(Array3D &&other) noexcept : data_(other.data_), shape_(other.shape_) {
    other.data_ = nullptr;
    other.shape_ = std::array<int, 3>{-1, -1, -1};
  }
  Array3D &operator=(Array3D &&other) noexcept {
    if (this != &other) {
      if (data_)
        cudaFree(data_);
      data_ = other.data_;
      shape_ = other.shape_;
      other.data_ = nullptr;
      other.shape_ = std::array<int, 3>{-1, -1, -1};
    }
    return *this;
  }
};
#endif  // USE_CUDA

}  // namespace miso
