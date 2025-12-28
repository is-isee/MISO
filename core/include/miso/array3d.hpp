#pragma once

#include <algorithm>
#include <cassert>

#include <miso/cuda_compat.hpp>
#include <miso/memory_space.hpp>
#ifdef USE_CUDA
#include <miso/cuda_util.cuh>
#endif  // USE_CUDA

namespace miso {

/// @brief Lightweight non-owning view of MHD fields.
template <typename T> class Array3DView {
private:
  T *data_ = nullptr;
  int i_total_ = -1, j_total_ = -1, k_total_ = -1;

public:
  __host__ __device__ Array3DView(T *data, int i_total, int j_total,
                                  int k_total) noexcept
      : data_(data), i_total_(i_total), j_total_(j_total), k_total_(k_total) {}

  __host__ __device__ T &operator()(int i, int j, int k) noexcept {
    assert(i >= 0 && i < i_total_);
    assert(j >= 0 && j < j_total_);
    assert(k >= 0 && k < k_total_);
    return data_[i * j_total_ * k_total_ + j * k_total_ + k];
  }
  __host__ __device__ const T &operator()(int i, int j, int k) const noexcept {
    assert(i >= 0 && i < i_total_);
    assert(j >= 0 && j < j_total_);
    assert(k >= 0 && k < k_total_);
    return data_[i * j_total_ * k_total_ + j * k_total_ + k];
  }

  __host__ __device__ T &operator[](int idx) noexcept {
    assert(idx >= 0 && idx < i_total_ * j_total_ * k_total_);
    return data_[idx];
  }
  __host__ __device__ const T &operator[](int idx) const noexcept {
    assert(idx >= 0 && idx < i_total_ * j_total_ * k_total_);
    return data_[idx];
  }

  __host__ __device__ const T *data() const noexcept { return data_; }

  __host__ __device__ int size_x() const noexcept { return i_total_; }
  __host__ __device__ int size_y() const noexcept { return j_total_; }
  __host__ __device__ int size_z() const noexcept { return k_total_; }
  __host__ __device__ size_t size() const noexcept {
    return i_total_ * j_total_ * k_total_;
  }
};

/// @brief 3D Array in general memory space.
template <typename T, typename MemorySpace = HostSpace> class Array3D;

/// @brief 3D Array in host memory.
template <typename T> class Array3D<T, HostSpace> {
private:
  T *data_ = nullptr;
  int i_total_ = -1, j_total_ = -1, k_total_ = -1;

public:
  Array3D(int i_total, int j_total, int k_total)
      : i_total_(i_total), j_total_(j_total), k_total_(k_total) {
    data_ = new T[i_total * j_total * k_total];
  }

  ~Array3D() { delete[] data_; }

  /// @brief Get a lightweight view of the array.
  Array3DView<T> view() noexcept {
    return Array3DView<T>(data_, i_total_, j_total_, k_total_);
  }

  /// @brief Get a constant lightweight view of the array.
  Array3DView<const T> view() const noexcept {
    return Array3DView<const T>(data_, i_total_, j_total_, k_total_);
  }

  T &operator()(int i, int j, int k) noexcept {
    assert(i >= 0 && i < i_total_);
    assert(j >= 0 && j < j_total_);
    assert(k >= 0 && k < k_total_);
    return data_[i * j_total_ * k_total_ + j * k_total_ + k];
  }
  const T &operator()(int i, int j, int k) const noexcept {
    assert(i >= 0 && i < i_total_);
    assert(j >= 0 && j < j_total_);
    assert(k >= 0 && k < k_total_);
    return data_[i * j_total_ * k_total_ + j * k_total_ + k];
  }

  T &operator[](int idx) noexcept {
    assert(idx >= 0 && idx < i_total_ * j_total_ * k_total_);
    return data_[idx];
  }
  const T &operator[](int idx) const noexcept {
    assert(idx >= 0 && idx < i_total_ * j_total_ * k_total_);
    return data_[idx];
  }

  /// @brief Get pointer to the data.
  T *data() noexcept { return data_; }

  /// @brief Get const pointer to the data.
  const T *data() const noexcept { return data_; }

  /// @brief Get size in x direction.
  int size_x() const noexcept { return i_total_; }

  /// @brief Get size in y direction.
  int size_y() const noexcept { return j_total_; }

  /// @brief Get size in z direction.
  int size_z() const noexcept { return k_total_; }

  /// @brief Get total size of the array.
  size_t size() const noexcept { return i_total_ * j_total_ * k_total_; }

  /// @brief Copy data from another Array3D in host memory.
  void copy_from(const Array3D<T, HostSpace> &other) {
    assert(other.size_x() == size_x());
    assert(other.size_y() == size_y());
    assert(other.size_z() == size_z());
    assert(other.data() && data());
    std::copy(other.data(), other.data() + other.size(), data());
  }

#ifdef USE_CUDA
  /// @brief Copy data from another Array3D in CUDA memory.
  void copy_from(const Array3D<T, CUDASpace> &other) {
    assert(other.size_x() == size_x());
    assert(other.size_y() == size_y());
    assert(other.size_z() == size_z());
    assert(other.data() && data());
    MISO_CUDA_CHECK(cudaMemcpy(data(), other.data(), sizeof(T) * other.size(),
                               cudaMemcpyDeviceToHost));
  }
#endif  // USE_CUDA

  // Prohibit copy and move semantics
  Array3D(const Array3D &) = delete;
  Array3D &operator=(const Array3D &) = delete;
  Array3D(Array3D &&) = delete;
  Array3D &operator=(Array3D &&) = delete;
};

#ifdef USE_CUDA
/// @brief 3D Array in CUDA device memory.
template <typename T> class Array3D<T, CUDASpace> {
private:
  T *data_ = nullptr;
  int i_total_ = -1, j_total_ = -1, k_total_ = -1;

public:
  Array3D(int i_total, int j_total, int k_total)
      : i_total_(i_total), j_total_(j_total), k_total_(k_total) {
    const auto array_size = sizeof(T) * i_total_ * j_total_ * k_total_;
    MISO_CUDA_CHECK(cudaMalloc(&data_, array_size));
  }

  ~Array3D() {
    if (data_)
      cudaFree(data_);
    data_ = nullptr;
  }

  /// @brief Get a lightweight view of the array.
  Array3DView<T> view() noexcept {
    return Array3DView<T>(data_, i_total_, j_total_, k_total_);
  }

  /// @brief Get a constant lightweight view of the array.
  Array3DView<const T> view() const noexcept {
    return Array3DView<const T>(data_, i_total_, j_total_, k_total_);
  }

  /// @brief Get pointer to the data.
  T *data() noexcept { return data_; }

  /// @brief Get const pointer to the data.
  const T *data() const noexcept { return data_; }

  /// @brief Get size in x direction.
  int size_x() const noexcept { return i_total_; }

  /// @brief Get size in y direction.
  int size_y() const noexcept { return j_total_; }

  /// @brief Get size in z direction.
  int size_z() const noexcept { return k_total_; }

  /// @brief Get total size of the array.
  size_t size() const noexcept { return i_total_ * j_total_ * k_total_; }

  /// @brief Copy data from another Array3D in host memory.
  void copy_from(const Array3D<T, HostSpace> &other) {
    assert(other.size_x() == size_x());
    assert(other.size_y() == size_y());
    assert(other.size_z() == size_z());
    assert(other.data() && data());
    MISO_CUDA_CHECK(cudaMemcpy(data(), other.data(), sizeof(T) * size(),
                               cudaMemcpyHostToDevice));
  }

  /// @brief Copy data from another Array3D in CUDA memory.
  void copy_from(const Array3D<T, CUDASpace> &other) {
    assert(other.size_x() == size_x());
    assert(other.size_y() == size_y());
    assert(other.size_z() == size_z());
    assert(other.data() && data());
    MISO_CUDA_CHECK(cudaMemcpy(data(), other.data(), sizeof(T) * size(),
                               cudaMemcpyDeviceToDevice));
  }

  // Prohibit copy and move semantics
  Array3D(const Array3D &) = delete;
  Array3D &operator=(const Array3D &) = delete;
  Array3D(Array3D &&) = delete;
  Array3D &operator=(Array3D &&) = delete;
};
#endif  // USE_CUDA

}  // namespace miso
