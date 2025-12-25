#pragma once

#include <miso/cuda_compat.hpp>

namespace miso {
namespace mhd {

/// @brief Lightweight non-owning view of MHD fields.
template <typename Real> struct FieldsView {
  // Real *data_ro = nullptr;
  // Real *data_vx = nullptr;
  // Real *data_vy = nullptr;
  // Real *data_vz = nullptr;
  // Real *data_bx = nullptr;
  // Real *data_by = nullptr;
  // Real *data_bz = nullptr;
  // Real *data_ei = nullptr;
  // Real *data_ph = nullptr;
  Real *ro = nullptr;
  Real *vx = nullptr;
  Real *vy = nullptr;
  Real *vz = nullptr;
  Real *bx = nullptr;
  Real *by = nullptr;
  Real *bz = nullptr;
  Real *ei = nullptr;
  Real *ph = nullptr;
  int i_total = -1, j_total = -1, k_total = -1;

  // template <typename FieldsType>
  // HOST_DEVICE explicit FieldsView(FieldsType &fields) noexcept
  //     : data_ro(fields.data_ro), data_vx(fields.data_vx), data_vy(fields.data_vy),
  //       data_vz(fields.data_vz), data_bx(fields.data_bx), data_by(fields.data_by),
  //       data_bz(fields.data_bz), data_ei(fields.data_ei), data_ph(fields.data_ph),
  //       i_total(fields.i_total), j_total(fields.j_total),
  //       k_total(fields.k_total)  {}

  HOST_DEVICE
  FieldsView(Real *ro, Real *vx, Real *vy, Real *vz, Real *bx, Real *by, Real *bz,
             Real *ei, Real *ph, int i_total, int j_total, int k_total) noexcept
      : ro(ro), vx(vx), vy(vy), vz(vz), bx(bx), by(by), bz(bz), ei(ei), ph(ph),
        i_total(i_total), j_total(j_total), k_total(k_total) {}

  template <typename FieldsType>
  HOST_DEVICE explicit FieldsView(FieldsType &fields) noexcept
      : ro(fields.ro), vx(fields.vx), vy(fields.vy), vz(fields.vz), bx(fields.bx),
        by(fields.by), bz(fields.bz), ei(fields.ei), ph(fields.ph),
        i_total(fields.i_total), j_total(fields.j_total),
        k_total(fields.k_total) {}

  // // Shallow-const non-owning array handles.
  // HOST_DEVICE Real &ro(int i, int j, int k) const noexcept {
  //   return data_ro[i * j_total * k_total + j * k_total + k];
  // }
  // HOST_DEVICE Real &vx(int i, int j, int k) const noexcept {
  //   return data_vx[i * j_total * k_total + j * k_total + k];
  // }
  // HOST_DEVICE Real &vy(int i, int j, int k) const noexcept {
  //   return data_vy[i * j_total * k_total + j * k_total + k];
  // }
  // HOST_DEVICE Real &vz(int i, int j, int k) const noexcept {
  //   return data_vz[i * j_total * k_total + j * k_total + k];
  // }
  // HOST_DEVICE Real &bx(int i, int j, int k) const noexcept {
  //   return data_bx[i * j_total * k_total + j * k_total + k];
  // }
  // HOST_DEVICE Real &by(int i, int j, int k) const noexcept {
  //   return data_by[i * j_total * k_total + j * k_total + k];
  // }
  // HOST_DEVICE Real &bz(int i, int j, int k) const noexcept {
  //   return data_bz[i * j_total * k_total + j * k_total + k];
  // }
  // HOST_DEVICE Real &ei(int i, int j, int k) const noexcept {
  //   return data_ei[i * j_total * k_total + j * k_total + k];
  // }
  // HOST_DEVICE Real &ph(int i, int j, int k) const noexcept {
  //   return data_ph[i * j_total * k_total + j * k_total + k];
  // }

  HOST_DEVICE int size_x() const noexcept { return i_total; }
  HOST_DEVICE int size_y() const noexcept { return j_total; }
  HOST_DEVICE int size_z() const noexcept { return k_total; }
  HOST_DEVICE int size() const noexcept { return i_total * j_total * k_total; }
};

}  // namespace mhd
}  // namespace miso
