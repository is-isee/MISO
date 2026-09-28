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

// Source terms are optional. A Source type may define any of the following
// member functions, each returning the source term of the equation for the
// named variable at cell (i, j, k):
//   Real ro(FieldsView<const Real> qq, int i, int j, int k) const;  // mass
//   Real vx(...), vy(...), vz(...)  // momentum (force per unit volume)
//   Real bx(...), by(...), bz(...)  // induction equation
//   Real ei(...)                    // total energy (per unit volume and time)
// A term that is not defined is not computed (treated as 0).
// clang-format off
#define MISO_MHD_DEFINE_SOURCE_TERM(name)                                      \
  namespace impl_source {                                                      \
  template <class Source, class Real, class = void>                            \
  struct has_##name : std::false_type {};                                      \
  template <class Source, class Real>                                          \
  struct has_##name<Source, Real,                                              \
                    std::void_t<decltype(std::declval<const Source &>().name(  \
                        std::declval<FieldsView<const Real>>(), 0, 0, 0))>>    \
      : std::true_type {};                                                     \
  }                                                                            \
  /** @brief Source term of `name` (0 if the Source does not define it) */     \
  template <typename Real, typename Source>                                    \
  __host__ __device__ inline Real source_##name(                               \
      const Source &src, const FieldsView<const Real> &qq, int i, int j,       \
      int k) {                                                                 \
    if constexpr (impl_source::has_##name<Source, Real>::value) {              \
      return src.name(qq, i, j, k);                                            \
    } else {                                                                   \
      return Real(0);                                                          \
    }                                                                          \
  }
MISO_MHD_DEFINE_SOURCE_TERM(ro)
MISO_MHD_DEFINE_SOURCE_TERM(vx)
MISO_MHD_DEFINE_SOURCE_TERM(vy)
MISO_MHD_DEFINE_SOURCE_TERM(vz)
MISO_MHD_DEFINE_SOURCE_TERM(bx)
MISO_MHD_DEFINE_SOURCE_TERM(by)
MISO_MHD_DEFINE_SOURCE_TERM(bz)
MISO_MHD_DEFINE_SOURCE_TERM(ei)
#undef MISO_MHD_DEFINE_SOURCE_TERM
// clang-format on

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
