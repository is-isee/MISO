#pragma once

#include "array3d.hpp"
#include "backend.hpp"
#ifdef __CUDACC__
#include "cuda_util.cuh"
#endif  // __CUDACC__

namespace miso {
namespace mhd {

/// @brief Primitive MHD variables
template <typename Real, typename Backend = backend::Host> struct Fields;

/// @brief Lightweight non-owning view of MHD fields.
template <typename T>  // Use T to represent Real or const Real
struct FieldsView {
  Array3DView<T> ro, vx, vy, vz, bx, by, bz, ei, ph;

  explicit FieldsView(Array3DView<T> ro_, Array3DView<T> vx_, Array3DView<T> vy_,
                      Array3DView<T> vz_, Array3DView<T> bx_, Array3DView<T> by_,
                      Array3DView<T> bz_, Array3DView<T> ei_,
                      Array3DView<T> ph_) noexcept
      : ro(ro_), vx(vx_), vy(vy_), vz(vz_), bx(bx_), by(by_), bz(bz_), ei(ei_),
        ph(ph_) {}

  /// @brief Get extent of specified dimension
  __host__ __device__ int extent(int dim) const noexcept {
    return ro.extent(dim);
  }

  /// @brief Get total size (number of elements)
  __host__ __device__ int size() const noexcept { return ro.size(); }
};

/// @brief Factory function to create FieldsView from Fields
template <typename Real, typename Backend>
FieldsView<Real> make_fields_view(Fields<Real, Backend> &fields) noexcept {
  return FieldsView<Real>(fields.ro.view(), fields.vx.view(), fields.vy.view(),
                          fields.vz.view(), fields.bx.view(), fields.by.view(),
                          fields.bz.view(), fields.ei.view(), fields.ph.view());
}

/// @brief Factory function to create const FieldsView from Fields
template <typename Real, typename Backend>
FieldsView<const Real>
make_fields_view(const Fields<Real, Backend> &fields) noexcept {
  return FieldsView<const Real>(
      fields.ro.view(), fields.vx.view(), fields.vy.view(), fields.vz.view(),
      fields.bx.view(), fields.by.view(), fields.bz.view(), fields.ei.view(),
      fields.ph.view());
}

/// @brief Primitive MHD variables on host
template <typename Real> struct Fields<Real, backend::Host> {
  Array3D<Real, backend::Host> ro, vx, vy, vz, bx, by, bz, ei, ph;

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

  FieldsView<Real> view() noexcept { return make_fields_view(*this); }
  FieldsView<const Real> view() const noexcept { return make_fields_view(*this); }
  FieldsView<const Real> const_view() const noexcept {
    return make_fields_view(*this);
  }

  void copy_from(const Fields<Real, backend::Host> &other) {
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

#ifdef __CUDACC__
  void copy_from(const Fields<Real, backend::CUDA> &other) {
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

  // Prohibit copy semantics
  Fields(const Fields &) = delete;
  Fields &operator=(const Fields &) = delete;

  // Allow move semantics
  Fields(Fields &&) = default;
  Fields &operator=(Fields &&) = default;
};

#ifdef __CUDACC__
/// @brief Primitive MHD variables on GPU device.
template <typename Real> struct Fields<Real, backend::CUDA> {
  Array3D<Real, backend::CUDA> ro, vx, vy, vz, bx, by, bz, ei, ph;

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

  FieldsView<Real> view() noexcept { return make_fields_view(*this); }
  FieldsView<const Real> view() const noexcept { return make_fields_view(*this); }
  FieldsView<const Real> const_view() const noexcept {
    return make_fields_view(*this);
  }
  void copy_from(const Fields<Real, backend::Host> &other) {
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

  void copy_from(const Fields<Real, backend::CUDA> &other) {
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

  // Prohibit copy semantics
  Fields(const Fields &) = delete;
  Fields &operator=(const Fields &) = delete;

  // Allow move semantics
  Fields(Fields &&) = default;
  Fields &operator=(Fields &&) = default;
};
#endif  // __CUDACC__

}  // namespace mhd
}  // namespace miso
