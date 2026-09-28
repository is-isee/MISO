#pragma once

#include <type_traits>
#include <utility>

#include "array3d.hpp"
#include "constants.hpp"
#include "cuda_compat.hpp"
#include "mhd_fields.hpp"

namespace miso {
namespace mhd {

template <typename Real, typename Backend> struct Integrator;

// Source terms of the induction equation are optional: a Source type may
// define bx, by, bz with the same signature as vx. They are 0 otherwise.
namespace impl_source {
template <class Source, class Real, class = void>
struct has_bx : std::false_type {};
template <class Source, class Real>
struct has_bx<Source, Real,
              std::void_t<decltype(std::declval<const Source &>().bx(
                  std::declval<FieldsView<const Real>>(), 0, 0, 0))>>
    : std::true_type {};

template <class Source, class Real, class = void>
struct has_by : std::false_type {};
template <class Source, class Real>
struct has_by<Source, Real,
              std::void_t<decltype(std::declval<const Source &>().by(
                  std::declval<FieldsView<const Real>>(), 0, 0, 0))>>
    : std::true_type {};

template <class Source, class Real, class = void>
struct has_bz : std::false_type {};
template <class Source, class Real>
struct has_bz<Source, Real,
              std::void_t<decltype(std::declval<const Source &>().bz(
                  std::declval<FieldsView<const Real>>(), 0, 0, 0))>>
    : std::true_type {};
}  // namespace impl_source

/// @brief Source term of the x induction equation (0 if not defined)
template <typename Real, typename Source>
__host__ __device__ inline Real source_bx(const Source &src,
                                          const FieldsView<const Real> &qq, int i,
                                          int j, int k) {
  if constexpr (impl_source::has_bx<Source, Real>::value) {
    return src.bx(qq, i, j, k);
  } else {
    return Real(0);
  }
}

/// @brief Source term of the y induction equation (0 if not defined)
template <typename Real, typename Source>
__host__ __device__ inline Real source_by(const Source &src,
                                          const FieldsView<const Real> &qq, int i,
                                          int j, int k) {
  if constexpr (impl_source::has_by<Source, Real>::value) {
    return src.by(qq, i, j, k);
  } else {
    return Real(0);
  }
}

/// @brief Source term of the z induction equation (0 if not defined)
template <typename Real, typename Source>
__host__ __device__ inline Real source_bz(const Source &src,
                                          const FieldsView<const Real> &qq, int i,
                                          int j, int k) {
  if constexpr (impl_source::has_bz<Source, Real>::value) {
    return src.bz(qq, i, j, k);
  } else {
    return Real(0);
  }
}

/// @brief Calculate 4th order space-centered derivative for qq
template <typename Real>
__host__ __device__ inline Real
space_centered_4th(Array3DView<const Real> qq, Real dxyzi, int i, int j, int k,
                   int is, int js, int ks) {
  // clang-format off
  return (
    -     qq(i + 2*is, j + 2*js, k + 2*ks)
    + 8.0*qq(i +   is, j +   js, k +   ks)
    - 8.0*qq(i -   is, j -   js, k -   ks)
    +     qq(i - 2*is, j - 2*js, k - 2*ks)
  )*inv12<Real>*dxyzi;
  // clang-format on
};

/// @brief Calculate 4th order space-centered derivative for qq1*qq2
template <typename Real>
__host__ __device__ inline Real
space_centered_4th(Array3DView<const Real> qq1, Array3DView<const Real> qq2,
                   Real dxyzi, int i, int j, int k, int is, int js, int ks) {
  // clang-format off
  return (
    -     qq1(i + 2*is, j + 2*js, k + 2*ks)*qq2(i + 2*is, j + 2*js, k + 2*ks)
    + 8.0*qq1(i +   is, j +   js, k +   ks)*qq2(i +   is, j +   js, k +   ks)
    - 8.0*qq1(i -   is, j -   js, k -   ks)*qq2(i -   is, j -   js, k -   ks)
    +     qq1(i - 2*is, j - 2*js, k - 2*ks)*qq2(i - 2*is, j - 2*js, k - 2*ks)
  )*inv12<Real>*dxyzi;
  // clang-format on
};

/// @brief Calculate 4th order space-centered derivative for qq1*qq2*qq3
template <typename Real>
__host__ __device__ inline Real
space_centered_4th(Array3DView<const Real> qq1, Array3DView<const Real> qq2,
                   Array3DView<const Real> qq3, Real dxyzi, int i, int j, int k,
                   int is, int js, int ks) {
  // clang-format off
  return (
    -     qq1(i + 2*is, j + 2*js, k + 2*ks)*qq2(i + 2*is, j + 2*js, k + 2*ks)*qq3(i + 2*is, j + 2*js, k + 2*ks)
    + 8.0*qq1(i +   is, j +   js, k +   ks)*qq2(i +   is, j +   js, k +   ks)*qq3(i +   is, j +   js, k +   ks)
    - 8.0*qq1(i -   is, j -   js, k -   ks)*qq2(i -   is, j -   js, k -   ks)*qq3(i -   is, j -   js, k -   ks)
    +     qq1(i - 2*is, j - 2*js, k - 2*ks)*qq2(i - 2*is, j - 2*js, k - 2*ks)*qq3(i - 2*is, j - 2*js, k - 2*ks)
  )*inv12<Real>*dxyzi;
  // clang-format on
};

}  // namespace mhd
}  // namespace miso

#include "mhd_integrator_host.hpp"

#ifdef __CUDACC__
#include "mhd_integrator_cuda.cuh"
#endif
