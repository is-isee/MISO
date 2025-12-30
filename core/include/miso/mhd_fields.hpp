#pragma once

#include <miso/array3d.hpp>
#include <miso/policy.hpp>
#ifdef USE_CUDA
#include <miso/cuda_util.cuh>
#endif  // USE_CUDA

namespace miso {
namespace mhd {

/// @brief Lightweight non-owning view of MHD fields.
template <typename T>  // Use T to represent Real or const Real
struct FieldsView {
  Array3DView<T> ro, vx, vy, vz, bx, by, bz, ei, ph;

  template <typename FieldType>
  __host__ __device__ explicit FieldsView(FieldType &fields) noexcept
      : ro(fields.ro.view()), vx(fields.vx.view()), vy(fields.vy.view()),
        vz(fields.vz.view()), bx(fields.bx.view()), by(fields.by.view()),
        bz(fields.bz.view()), ei(fields.ei.view()), ph(fields.ph.view()) {}

  __host__ __device__ int size_x() const noexcept { return ro.size_x(); }
  __host__ __device__ int size_y() const noexcept { return ro.size_y(); }
  __host__ __device__ int size_z() const noexcept { return ro.size_z(); }
  __host__ __device__ size_t size() const noexcept { return ro.size(); }
};

/// @brief Primitive MHD variables
template <typename Real, typename Space = HostSpace> struct Fields;

/// @brief Primitive MHD variables on host
template <typename Real> struct Fields<Real, HostSpace> {
  Array3D<Real, HostSpace> ro, vx, vy, vz, bx, by, bz, ei, ph;

  Fields(const int i_total, const int j_total, const int k_total)
      : ro(i_total, j_total, k_total), vx(i_total, j_total, k_total),
        vy(i_total, j_total, k_total), vz(i_total, j_total, k_total),
        bx(i_total, j_total, k_total), by(i_total, j_total, k_total),
        bz(i_total, j_total, k_total), ei(i_total, j_total, k_total),
        ph(i_total, j_total, k_total) {}

  template <typename GridType>
  explicit Fields(const GridType &grid)
      : ro(grid.i_total, grid.j_total, grid.k_total),
        vx(grid.i_total, grid.j_total, grid.k_total),
        vy(grid.i_total, grid.j_total, grid.k_total),
        vz(grid.i_total, grid.j_total, grid.k_total),
        bx(grid.i_total, grid.j_total, grid.k_total),
        by(grid.i_total, grid.j_total, grid.k_total),
        bz(grid.i_total, grid.j_total, grid.k_total),
        ei(grid.i_total, grid.j_total, grid.k_total),
        ph(grid.i_total, grid.j_total, grid.k_total) {}

  FieldsView<const Real> view() const noexcept {
    return FieldsView<const Real>(*this);
  }
  FieldsView<Real> view() noexcept { return FieldsView<Real>(*this); }

  void copy_from(const Fields<Real, HostSpace> &other) {
    ro.copy_from(other.ro);
    vx.copy_from(other.vx);
    vy.copy_from(other.vy);
    vz.copy_from(other.vz);
    bx.copy_from(other.bx);
    by.copy_from(other.by);
    bz.copy_from(other.bz);
    ei.copy_from(other.ei);
    ph.copy_from(other.ph);
  }

#ifdef USE_CUDA
  void copy_from(const Fields<Real, CUDASpace> &other) {
    ro.copy_from(other.ro);
    vx.copy_from(other.vx);
    vy.copy_from(other.vy);
    vz.copy_from(other.vz);
    bx.copy_from(other.bx);
    by.copy_from(other.by);
    bz.copy_from(other.bz);
    ei.copy_from(other.ei);
    ph.copy_from(other.ph);
    cudaDeviceSynchronize();
  }
#endif

  // Prohibit copy and move semantics
  Fields(const Fields &) = delete;
  Fields &operator=(const Fields &) = delete;
  Fields(Fields &&) = delete;
  Fields &operator=(Fields &&) = delete;
};

#ifdef USE_CUDA
/// @brief Primitive MHD variables on GPU device.
template <typename Real> struct Fields<Real, CUDASpace> {
  Array3D<Real, CUDASpace> ro, vx, vy, vz, bx, by, bz, ei, ph;

  Fields(const int i_total, const int j_total, const int k_total)
      : ro(i_total, j_total, k_total), vx(i_total, j_total, k_total),
        vy(i_total, j_total, k_total), vz(i_total, j_total, k_total),
        bx(i_total, j_total, k_total), by(i_total, j_total, k_total),
        bz(i_total, j_total, k_total), ei(i_total, j_total, k_total),
        ph(i_total, j_total, k_total) {}

  template <typename GridType>
  explicit Fields(const GridType &grid)
      : ro(grid.i_total, grid.j_total, grid.k_total),
        vx(grid.i_total, grid.j_total, grid.k_total),
        vy(grid.i_total, grid.j_total, grid.k_total),
        vz(grid.i_total, grid.j_total, grid.k_total),
        bx(grid.i_total, grid.j_total, grid.k_total),
        by(grid.i_total, grid.j_total, grid.k_total),
        bz(grid.i_total, grid.j_total, grid.k_total),
        ei(grid.i_total, grid.j_total, grid.k_total),
        ph(grid.i_total, grid.j_total, grid.k_total) {}

  __host__ __device__ FieldsView<const Real> view() const noexcept {
    return FieldsView<const Real>(*this);
  }
  __host__ __device__ FieldsView<Real> view() noexcept {
    return FieldsView<Real>(*this);
  }

  void copy_from(const Fields<Real, HostSpace> &other) {
    ro.copy_from(other.ro);
    vx.copy_from(other.vx);
    vy.copy_from(other.vy);
    vz.copy_from(other.vz);
    bx.copy_from(other.bx);
    by.copy_from(other.by);
    bz.copy_from(other.bz);
    ei.copy_from(other.ei);
    ph.copy_from(other.ph);
    cudaDeviceSynchronize();
  }

  void copy_from(const Fields<Real, CUDASpace> &other) {
    ro.copy_from(other.ro);
    vx.copy_from(other.vx);
    vy.copy_from(other.vy);
    vz.copy_from(other.vz);
    bx.copy_from(other.bx);
    by.copy_from(other.by);
    bz.copy_from(other.bz);
    ei.copy_from(other.ei);
    ph.copy_from(other.ph);
    cudaDeviceSynchronize();
  }

  // Prohibit copy and move semantics
  Fields(const Fields &) = delete;
  Fields &operator=(const Fields &) = delete;
  Fields(Fields &&) = delete;
  Fields &operator=(Fields &&) = delete;
};
#endif  // USE_CUDA

}  // namespace mhd
}  // namespace miso
