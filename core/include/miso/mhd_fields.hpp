#pragma once

#include <miso/array3d.hpp>
#include <miso/backend.hpp>
#ifdef __CUDACC__
#include <miso/cuda_util.cuh>
#endif  // __CUDACC__

namespace miso {
namespace mhd {

/// @brief Primitive MHD variables
template <typename Real, typename Backend = backend::Host> struct Fields;

/// @brief Lightweight non-owning view of MHD fields.
template <typename T>  // Use T to represent Real or const Real
class FieldsView {
private:
  // Always constructed from Fields.
  template <typename, typename> friend struct Fields;
  template <typename FieldsType>
  explicit FieldsView(FieldsType &fields) noexcept
      : ro(fields.ro.view()), vx(fields.vx.view()), vy(fields.vy.view()),
        vz(fields.vz.view()), bx(fields.bx.view()), by(fields.by.view()),
        bz(fields.bz.view()), ei(fields.ei.view()), ph(fields.ph.view()) {}

public:
  Array3DView<T> ro, vx, vy, vz, bx, by, bz, ei, ph;

  __host__ __device__ int extent(int dim) const noexcept {
    return ro.extent(dim);
  }
  __host__ __device__ std::array<int, 3> shape() const noexcept {
    return ro.shape();
  }
  __host__ __device__ int size() const noexcept { return ro.size(); }
};

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

  FieldsView<Real> view() noexcept { return FieldsView<Real>(*this); }
  FieldsView<const Real> view() const noexcept {
    return FieldsView<const Real>(*this);
  }
  FieldsView<const Real> const_view() const noexcept { return view(); }

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

  FieldsView<Real> view() noexcept { return FieldsView<Real>(*this); }
  FieldsView<const Real> view() const noexcept {
    return FieldsView<const Real>(*this);
  }
  FieldsView<const Real> const_view() const noexcept { return view(); }

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
