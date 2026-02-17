#pragma once

#include "array3d.hpp"
#include "constants.hpp"
#include "cuda_compat.hpp"
#include "mhd_fields.hpp"

namespace miso {
namespace mhd {

template <typename Real, typename Backend> struct Integrator;

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
