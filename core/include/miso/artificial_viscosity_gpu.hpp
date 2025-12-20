#pragma once

#include <algorithm>
#include <cassert>
#include <initializer_list>
#include <string>

#include <miso/artificial_viscosity_core.hpp>
#include <miso/constants.hpp>
#include <miso/cuda_utils.cuh>
#include <miso/grid_gpu.cuh>
#include <miso/model.hpp>

namespace miso {
namespace mhd {
namespace artificial_viscosity {

template <typename Real>
__global__ void characteristic_velocity_eval_kernel(Array3DDevice<Real> cc_d,
                                                    MHDCoreDevice<Real> qq,
                                                    GridDevice<Real> grid,
                                                    Real eos_gm, Real cs_fac,
                                                    Real ca_fac, Real vv_fac) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i <= grid.i_total - 1 && j <= grid.j_total - 1 && k <= grid.k_total - 1) {
    // clang-format off
    Real cs = std::sqrt(eos_gm * (eos_gm - 1.0) * qq.ei[grid.idx(i, j, k)]);
    Real vv = std::sqrt(qq.vx[grid.idx(i, j, k)] * qq.vx[grid.idx(i, j, k)] +
                        qq.vy[grid.idx(i, j, k)] * qq.vy[grid.idx(i, j, k)] +
                        qq.vz[grid.idx(i, j, k)] * qq.vz[grid.idx(i, j, k)]);
    Real ca = std::sqrt((qq.bx[grid.idx(i, j, k)] * qq.bx[grid.idx(i, j, k)] +
                         qq.by[grid.idx(i, j, k)] * qq.by[grid.idx(i, j, k)] +
                         qq.bz[grid.idx(i, j, k)] * qq.bz[grid.idx(i, j, k)]) /
                         qq.ro[grid.idx(i, j, k)] * pii4<Real>);
    // clang-format on
    cc_d[grid.idx(i, j, k)] = cs * cs_fac + vv * vv_fac + ca * ca_fac;
  }
}

template <typename Real>
__global__ void
update_ro_kernel(MHDCoreDevice<Real> qq, MHDCoreDevice<Real> qq_rslt,
                 Array3DDevice<Real> cc, GridDevice<Real> grid, Real *dxyzi,
                 int i0_, int i1_, int j0_, int j1_, int k0_, int k1_, int is,
                 int js, int ks, Real ep, Real fh, Real dt) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= i0_ && i < i1_ && j >= j0_ && j < j1_ && k >= k0_ && k < k1_) {
    // chracteristic velocity
    // clang-format off
    Real ccl = cc[grid.idx(i - is, j - js, k - ks)];
    Real ccc = cc[grid.idx(i     , j     , k     )];
    Real ccr = cc[grid.idx(i + is, j + js, k + ks)];

    // density
    Real qql2 = qq.ro[grid.idx(i - 2 * is, j - 2 * js, k - 2 * ks)];
    Real qql1 = qq.ro[grid.idx(i -     is, j -     js, k -     ks)];
    Real qqc  = qq.ro[grid.idx(i         , j         , k         )];
    Real qqr1 = qq.ro[grid.idx(i +     is, j +     js, k +     ks)];
    Real qqr2 = qq.ro[grid.idx(i + 2 * is, j + 2 * js, k + 2 * ks)];
    // clang-format on

    // dqq at i-is, j-js, k-2ks
    Real dqq_dw = artificial_viscosity::dqq_eval(qql2, qql1, qqc, ep);
    // dqq at i, j, k
    Real dqq_cn = artificial_viscosity::dqq_eval(qql1, qqc, qqr1, ep);
    // dqq at i+is, j+js, k+ks
    Real dqq_up = artificial_viscosity::dqq_eval(qqc, qqr1, qqr2, ep);
    Real fro_dw = artificial_viscosity::flux_core(qql1, qqc, dqq_dw, dqq_cn,
                                                  Real(0.5) * (ccl + ccc), fh);
    Real fro_up = artificial_viscosity::flux_core(qqc, qqr1, dqq_cn, dqq_up,
                                                  Real(0.5) * (ccc + ccr), fh);

    qq_rslt.ro[grid.idx(i, j, k)] =
        qq.ro[grid.idx(i, j, k)] -
        (fro_up - fro_dw) * dxyzi[i * is + j * js + k * ks] * dt;
  }
}

template <typename Real>
__global__ void
update_vx_kernel(MHDCoreDevice<Real> qq, MHDCoreDevice<Real> qq_rslt,
                 Array3DDevice<Real> cc, GridDevice<Real> grid, Real *dxyzi,
                 int i0_, int i1_, int j0_, int j1_, int k0_, int k1_, int is,
                 int js, int ks, Real ep, Real fh, Real dt) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= i0_ && i < i1_ && j >= j0_ && j < j1_ && k >= k0_ && k < k1_) {
    // chracteristic velocity
    // clang-format off
    Real ccl = cc[grid.idx(i - is, j - js, k - ks)];
    Real ccc = cc[grid.idx(i     , j     , k     )];
    Real ccr = cc[grid.idx(i + is, j + js, k + ks)];

    // x momentum
    Real qql2 = qq.ro[grid.idx(i - 2 * is, j - 2 * js, k - 2 * ks)] *
                qq.vx[grid.idx(i - 2 * is, j - 2 * js, k - 2 * ks)];
    Real qql1 = qq.ro[grid.idx(i -     is, j -     js, k -     ks)] *
                qq.vx[grid.idx(i -     is, j -     js, k -     ks)];
    Real qqc  = qq.ro[grid.idx(i         , j         , k         )] *
                qq.vx[grid.idx(i         , j         , k         )];
    Real qqr1 = qq.ro[grid.idx(i +     is, j +     js, k +     ks)] *
                qq.vx[grid.idx(i +     is, j +     js, k +     ks)];
    Real qqr2 = qq.ro[grid.idx(i + 2 * is, j + 2 * js, k + 2 * ks)] *
                qq.vx[grid.idx(i + 2 * is, j + 2 * js, k + 2 * ks)];
    // clang-format on
    // dqq at i-is, j-js, k-2ks
    Real dqq_dw = artificial_viscosity::dqq_eval(qql2, qql1, qqc, ep);
    // dqq at i, j, k
    Real dqq_cn = artificial_viscosity::dqq_eval(qql1, qqc, qqr1, ep);
    // dqq at i+is, j+js, k+ks
    Real dqq_up = artificial_viscosity::dqq_eval(qqc, qqr1, qqr2, ep);
    Real frx_dw = artificial_viscosity::flux_core(qql1, qqc, dqq_dw, dqq_cn,
                                                  Real(0.5) * (ccl + ccc), fh);
    Real frx_up = artificial_viscosity::flux_core(qqc, qqr1, dqq_cn, dqq_up,
                                                  Real(0.5) * (ccc + ccr), fh);

    qq_rslt.vx[grid.idx(i, j, k)] =
        (qq.ro[grid.idx(i, j, k)] * qq.vx[grid.idx(i, j, k)] -
         (frx_up - frx_dw) * dxyzi[i * is + j * js + k * ks] * dt) /
        qq_rslt.ro[grid.idx(i, j, k)];
  }
}

template <typename Real>
__global__ void
update_vy_kernel(MHDCoreDevice<Real> qq, MHDCoreDevice<Real> qq_rslt,
                 Array3DDevice<Real> cc, GridDevice<Real> grid, Real *dxyzi,
                 int i0_, int i1_, int j0_, int j1_, int k0_, int k1_, int is,
                 int js, int ks, Real ep, Real fh, Real dt) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= i0_ && i < i1_ && j >= j0_ && j < j1_ && k >= k0_ && k < k1_) {
    // chracteristic velocity
    // clang-format off
    Real ccl = cc[grid.idx(i - is, j - js, k - ks)];
    Real ccc = cc[grid.idx(i     , j     , k     )];
    Real ccr = cc[grid.idx(i + is, j + js, k + ks)];

    // x momentum
    Real qql2 = qq.ro[grid.idx(i - 2 * is, j - 2 * js, k - 2 * ks)] *
                qq.vy[grid.idx(i - 2 * is, j - 2 * js, k - 2 * ks)];
    Real qql1 = qq.ro[grid.idx(i -     is, j -     js, k -     ks)] *
                qq.vy[grid.idx(i -     is, j -     js, k -     ks)];
    Real qqc  = qq.ro[grid.idx(i         , j         , k         )] *
                qq.vy[grid.idx(i         , j         , k         )];
    Real qqr1 = qq.ro[grid.idx(i +     is, j +     js, k +     ks)] *
                qq.vy[grid.idx(i +     is, j +     js, k +     ks)];
    Real qqr2 = qq.ro[grid.idx(i + 2 * is, j + 2 * js, k + 2 * ks)] *
                qq.vy[grid.idx(i + 2 * is, j + 2 * js, k + 2 * ks)];
    // clang-format on
    // dqq at i-is, j-js, k-2ks
    Real dqq_dw = artificial_viscosity::dqq_eval(qql2, qql1, qqc, ep);
    // dqq at i, j, k
    Real dqq_cn = artificial_viscosity::dqq_eval(qql1, qqc, qqr1, ep);
    // dqq at i+is, j+js, k+ks
    Real dqq_up = artificial_viscosity::dqq_eval(qqc, qqr1, qqr2, ep);
    Real fry_dw = artificial_viscosity::flux_core(qql1, qqc, dqq_dw, dqq_cn,
                                                  Real(0.5) * (ccl + ccc), fh);
    Real fry_up = artificial_viscosity::flux_core(qqc, qqr1, dqq_cn, dqq_up,
                                                  Real(0.5) * (ccc + ccr), fh);

    qq_rslt.vy[grid.idx(i, j, k)] =
        (qq.ro[grid.idx(i, j, k)] * qq.vy[grid.idx(i, j, k)] -
         (fry_up - fry_dw) * dxyzi[i * is + j * js + k * ks] * dt) /
        qq_rslt.ro[grid.idx(i, j, k)];
  }
}

template <typename Real>
__global__ void
update_vz_kernel(MHDCoreDevice<Real> qq, MHDCoreDevice<Real> qq_rslt,
                 Array3DDevice<Real> cc, GridDevice<Real> grid, Real *dxyzi,
                 int i0_, int i1_, int j0_, int j1_, int k0_, int k1_, int is,
                 int js, int ks, Real ep, Real fh, Real dt) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= i0_ && i < i1_ && j >= j0_ && j < j1_ && k >= k0_ && k < k1_) {
    // chracteristic velocity
    // clang-format off
    Real ccl = cc[grid.idx(i - is, j - js, k - ks)];
    Real ccc = cc[grid.idx(i     , j     , k     )];
    Real ccr = cc[grid.idx(i + is, j + js, k + ks)];

    // x momentum
    Real qql2 = qq.ro[grid.idx(i - 2 * is, j - 2 * js, k - 2 * ks)] *
                qq.vz[grid.idx(i - 2 * is, j - 2 * js, k - 2 * ks)];
    Real qql1 = qq.ro[grid.idx(i -     is, j -     js, k -     ks)] *
                qq.vz[grid.idx(i -     is, j -     js, k -     ks)];
    Real qqc  = qq.ro[grid.idx(i         , j         , k         )] *
                qq.vz[grid.idx(i         , j         , k         )];
    Real qqr1 = qq.ro[grid.idx(i +     is, j +     js, k +     ks)] *
                qq.vz[grid.idx(i +     is, j +     js, k +     ks)];
    Real qqr2 = qq.ro[grid.idx(i + 2 * is, j + 2 * js, k + 2 * ks)] *
                qq.vz[grid.idx(i + 2 * is, j + 2 * js, k + 2 * ks)];
    // clang-format on
    // dqq at i-is, j-js, k-2ks
    Real dqq_dw = artificial_viscosity::dqq_eval(qql2, qql1, qqc, ep);
    // dqq at i, j, k
    Real dqq_cn = artificial_viscosity::dqq_eval(qql1, qqc, qqr1, ep);
    // dqq at i+is, j+js, k+ks
    Real dqq_up = artificial_viscosity::dqq_eval(qqc, qqr1, qqr2, ep);
    Real frz_dw = artificial_viscosity::flux_core(qql1, qqc, dqq_dw, dqq_cn,
                                                  Real(0.5) * (ccl + ccc), fh);
    Real frz_up = artificial_viscosity::flux_core(qqc, qqr1, dqq_cn, dqq_up,
                                                  Real(0.5) * (ccc + ccr), fh);

    qq_rslt.vz[grid.idx(i, j, k)] =
        (qq.ro[grid.idx(i, j, k)] * qq.vz[grid.idx(i, j, k)] -
         (frz_up - frz_dw) * dxyzi[i * is + j * js + k * ks] * dt) /
        qq_rslt.ro[grid.idx(i, j, k)];
  }
}

template <typename Real>
__global__ void
update_bx_kernel(MHDCoreDevice<Real> qq, MHDCoreDevice<Real> qq_rslt,
                 Array3DDevice<Real> cc, GridDevice<Real> grid, Real *dxyzi,
                 int i0_, int i1_, int j0_, int j1_, int k0_, int k1_, int is,
                 int js, int ks, Real ep, Real fh, Real dt) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= i0_ && i < i1_ && j >= j0_ && j < j1_ && k >= k0_ && k < k1_) {
    // chracteristic velocity
    // clang-format off
    Real ccl = cc[grid.idx(i - is, j - js, k - ks)];
    Real ccc = cc[grid.idx(i     , j     , k     )];
    Real ccr = cc[grid.idx(i + is, j + js, k + ks)];

    Real qql2 = qq.bx[grid.idx(i - 2 * is, j - 2 * js, k - 2 * ks)];
    Real qql1 = qq.bx[grid.idx(i -     is, j -     js, k -     ks)];
    Real qqc  = qq.bx[grid.idx(i         , j         , k         )];
    Real qqr1 = qq.bx[grid.idx(i +     is, j +     js, k +     ks)];
    Real qqr2 = qq.bx[grid.idx(i + 2 * is, j + 2 * js, k + 2 * ks)];
    // clang-format on
    // dqq at i-is, j-js, k-2ks
    Real dqq_dw = artificial_viscosity::dqq_eval(qql2, qql1, qqc, ep);
    // dqq at i, j, k
    Real dqq_cn = artificial_viscosity::dqq_eval(qql1, qqc, qqr1, ep);
    // dqq at i+is, j+js, k+ks
    Real dqq_up = artificial_viscosity::dqq_eval(qqc, qqr1, qqr2, ep);
    Real fbx_dw = artificial_viscosity::flux_core(qql1, qqc, dqq_dw, dqq_cn,
                                                  Real(0.5) * (ccl + ccc), fh);
    Real fbx_up = artificial_viscosity::flux_core(qqc, qqr1, dqq_cn, dqq_up,
                                                  Real(0.5) * (ccc + ccr), fh);

    qq_rslt.bx[grid.idx(i, j, k)] =
        qq.bx[grid.idx(i, j, k)] -
        (fbx_up - fbx_dw) * dxyzi[i * is + j * js + k * ks] * dt;
  }
}

template <typename Real>
__global__ void
update_by_kernel(MHDCoreDevice<Real> qq, MHDCoreDevice<Real> qq_rslt,
                 Array3DDevice<Real> cc, GridDevice<Real> grid, Real *dxyzi,
                 int i0_, int i1_, int j0_, int j1_, int k0_, int k1_, int is,
                 int js, int ks, Real ep, Real fh, Real dt) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= i0_ && i < i1_ && j >= j0_ && j < j1_ && k >= k0_ && k < k1_) {
    // chracteristic velocity
    // clang-format off
    Real ccl = cc[grid.idx(i - is, j - js, k - ks)];
    Real ccc = cc[grid.idx(i     , j     , k     )];
    Real ccr = cc[grid.idx(i + is, j + js, k + ks)];

    Real qql2 = qq.by[grid.idx(i - 2 * is, j - 2 * js, k - 2 * ks)];
    Real qql1 = qq.by[grid.idx(i -     is, j -     js, k -     ks)];
    Real qqc  = qq.by[grid.idx(i         , j         , k         )];
    Real qqr1 = qq.by[grid.idx(i +     is, j +     js, k +     ks)];
    Real qqr2 = qq.by[grid.idx(i + 2 * is, j + 2 * js, k + 2 * ks)];
    // clang-format on

    // dqq at i-is, j-js, k-2ks
    Real dqq_dw = artificial_viscosity::dqq_eval(qql2, qql1, qqc, ep);
    // dqq at i, j, k
    Real dqq_cn = artificial_viscosity::dqq_eval(qql1, qqc, qqr1, ep);
    // dqq at i+is, j+js, k+ks
    Real dqq_up = artificial_viscosity::dqq_eval(qqc, qqr1, qqr2, ep);
    Real fby_dw = artificial_viscosity::flux_core(qql1, qqc, dqq_dw, dqq_cn,
                                                  Real(0.5) * (ccl + ccc), fh);
    Real fby_up = artificial_viscosity::flux_core(qqc, qqr1, dqq_cn, dqq_up,
                                                  Real(0.5) * (ccc + ccr), fh);

    qq_rslt.by[grid.idx(i, j, k)] =
        qq.by[grid.idx(i, j, k)] -
        (fby_up - fby_dw) * dxyzi[i * is + j * js + k * ks] * dt;
  }
}

template <typename Real>
__global__ void
update_bz_kernel(MHDCoreDevice<Real> qq, MHDCoreDevice<Real> qq_rslt,
                 Array3DDevice<Real> cc, GridDevice<Real> grid, Real *dxyzi,
                 int i0_, int i1_, int j0_, int j1_, int k0_, int k1_, int is,
                 int js, int ks, Real ep, Real fh, Real dt) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= i0_ && i < i1_ && j >= j0_ && j < j1_ && k >= k0_ && k < k1_) {
    // chracteristic velocity
    // clang-format off
    Real ccl = cc[grid.idx(i - is, j - js, k - ks)];
    Real ccc = cc[grid.idx(i     , j     , k     )];
    Real ccr = cc[grid.idx(i + is, j + js, k + ks)];

    Real qql2 = qq.bz[grid.idx(i - 2 * is, j - 2 * js, k - 2 * ks)];
    Real qql1 = qq.bz[grid.idx(i -     is, j -     js, k -     ks)];
    Real qqc  = qq.bz[grid.idx(i         , j         , k         )];
    Real qqr1 = qq.bz[grid.idx(i +     is, j +     js, k +      ks)];
    Real qqr2 = qq.bz[grid.idx(i + 2 * is, j + 2 * js, k + 2 * ks)];
    // clang-format on

    // dqq at i-is, j-js, k-2ks
    Real dqq_dw = artificial_viscosity::dqq_eval(qql2, qql1, qqc, ep);
    // dqq at i, j, k
    Real dqq_cn = artificial_viscosity::dqq_eval(qql1, qqc, qqr1, ep);
    // dqq at i+is, j+js, k+ks
    Real dqq_up = artificial_viscosity::dqq_eval(qqc, qqr1, qqr2, ep);
    Real fbz_dw = artificial_viscosity::flux_core(qql1, qqc, dqq_dw, dqq_cn,
                                                  Real(0.5) * (ccl + ccc), fh);
    Real fbz_up = artificial_viscosity::flux_core(qqc, qqr1, dqq_cn, dqq_up,
                                                  Real(0.5) * (ccc + ccr), fh);

    qq_rslt.bz[grid.idx(i, j, k)] =
        qq.bz[grid.idx(i, j, k)] -
        (fbz_up - fbz_dw) * dxyzi[i * is + j * js + k * ks] * dt;
  }
}

template <typename Real>
__global__ void
update_ph_kernel(MHDCoreDevice<Real> qq, MHDCoreDevice<Real> qq_rslt,
                 Array3DDevice<Real> cc, GridDevice<Real> grid, Real *dxyzi,
                 int i0_, int i1_, int j0_, int j1_, int k0_, int k1_, int is,
                 int js, int ks, Real ep, Real fh, Real dt) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= i0_ && i < i1_ && j >= j0_ && j < j1_ && k >= k0_ && k < k1_) {
    // chracteristic velocity
    // clang-format off
    Real ccl = cc[grid.idx(i - is, j - js, k - ks)];
    Real ccc = cc[grid.idx(i     , j     , k     )];
    Real ccr = cc[grid.idx(i + is, j + js, k + ks)];

    Real qql2 = qq.ph[grid.idx(i - 2 * is, j - 2 * js, k - 2 * ks)];
    Real qql1 = qq.ph[grid.idx(i -     is, j -     js, k -     ks)];
    Real qqc  = qq.ph[grid.idx(i         , j         , k         )];
    Real qqr1 = qq.ph[grid.idx(i +     is, j +     js, k +     ks)];
    Real qqr2 = qq.ph[grid.idx(i + 2 * is, j + 2 * js, k + 2 * ks)];
    // clang-format on
    // dqq at i-is, j-js, k-2ks
    Real dqq_dw = artificial_viscosity::dqq_eval(qql2, qql1, qqc, ep);
    // dqq at i, j, k
    Real dqq_cn = artificial_viscosity::dqq_eval(qql1, qqc, qqr1, ep);
    // dqq at i+is, j+js, k+ks
    Real dqq_up = artificial_viscosity::dqq_eval(qqc, qqr1, qqr2, ep);
    Real fph_dw = artificial_viscosity::flux_core(qql1, qqc, dqq_dw, dqq_cn,
                                                  Real(0.5) * (ccl + ccc), fh);
    Real fph_up = artificial_viscosity::flux_core(qqc, qqr1, dqq_cn, dqq_up,
                                                  Real(0.5) * (ccc + ccr), fh);

    qq_rslt.ph[grid.idx(i, j, k)] =
        qq.ph[grid.idx(i, j, k)] -
        (fph_up - fph_dw) * dxyzi[i * is + j * js + k * ks] * dt;
  }
}

template <typename Real>
__global__ void
update_ei_kernel(MHDCoreDevice<Real> qq, MHDCoreDevice<Real> qq_rslt,
                 Array3DDevice<Real> cc, GridDevice<Real> grid, Real *dxyzi,
                 int i0_, int i1_, int j0_, int j1_, int k0_, int k1_, int is,
                 int js, int ks, Real ep, Real fh, Real dt) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= i0_ && i < i1_ && j >= j0_ && j < j1_ && k >= k0_ && k < k1_) {
    // chracteristic velocity
    // clang-format off
    Real ccl = cc[grid.idx(i - is, j - js, k - ks)];
    Real ccc = cc[grid.idx(i     , j     , k     )];
    Real ccr = cc[grid.idx(i + is, j + js, k + ks)];

    // total energy
    Real qql2 = qq.ro[grid.idx(i - 2 * is, j - 2 * js, k - 2 * ks)] *
                qq.ei[grid.idx(i - 2 * is, j - 2 * js, k - 2 * ks)];
    Real qql1 = qq.ro[grid.idx(i -     is, j -     js, k -     ks)] *
                qq.ei[grid.idx(i -     is, j -     js, k -     ks)];
    Real qqc  = qq.ro[grid.idx(i         , j         , k         )] *
                qq.ei[grid.idx(i         , j         , k         )];
    Real qqr1 = qq.ro[grid.idx(i +     is, j +     js, k +     ks)] *
                qq.ei[grid.idx(i +     is, j +     js, k +     ks)];
    Real qqr2 = qq.ro[grid.idx(i + 2 * is, j + 2 * js, k + 2 * ks)] *
                qq.ei[grid.idx(i + 2 * is, j + 2 * js, k + 2 * ks)];
    // clang-format on
    // dqq at i-is, j-js, k-2ks
    Real dqq_dw = artificial_viscosity::dqq_eval(qql2, qql1, qqc, ep);
    // dqq at i, j, k
    Real dqq_cn = artificial_viscosity::dqq_eval(qql1, qqc, qqr1, ep);
    // dqq at i+is, j+js, k+ks
    Real dqq_up = artificial_viscosity::dqq_eval(qqc, qqr1, qqr2, ep);
    Real fei_dw = artificial_viscosity::flux_core(qql1, qqc, dqq_dw, dqq_cn,
                                                  Real(0.5) * (ccl + ccc), fh);
    Real fei_up = artificial_viscosity::flux_core(qqc, qqr1, dqq_cn, dqq_up,
                                                  Real(0.5) * (ccc + ccr), fh);

    // Et: Total energy per unit volume, note that ei is internal energy per unit mass
    Real Et = qq.ro[grid.idx(i, j, k)] * qq.ei[grid.idx(i, j, k)] +
              0.5 * qq.ro[grid.idx(i, j, k)] *
                  (qq.vx[grid.idx(i, j, k)] * qq.vx[grid.idx(i, j, k)] +
                   qq.vy[grid.idx(i, j, k)] * qq.vy[grid.idx(i, j, k)] +
                   qq.vz[grid.idx(i, j, k)] * qq.vz[grid.idx(i, j, k)]) +
              pii8<Real> * (qq.bx[grid.idx(i, j, k)] * qq.bx[grid.idx(i, j, k)] +
                            qq.by[grid.idx(i, j, k)] * qq.by[grid.idx(i, j, k)] +
                            qq.bz[grid.idx(i, j, k)] * qq.bz[grid.idx(i, j, k)]);

    qq_rslt.ei[grid.idx(i, j, k)] =
        (Et - (fei_up - fei_dw) * dxyzi[i * is + j * js + k * ks] * dt -
         0.5 * qq_rslt.ro[grid.idx(i, j, k)] *
             (qq_rslt.vx[grid.idx(i, j, k)] * qq_rslt.vx[grid.idx(i, j, k)] +
              qq_rslt.vy[grid.idx(i, j, k)] * qq_rslt.vy[grid.idx(i, j, k)] +
              qq_rslt.vz[grid.idx(i, j, k)] * qq_rslt.vz[grid.idx(i, j, k)]) -
         pii8<Real> *
             (qq_rslt.bx[grid.idx(i, j, k)] * qq_rslt.bx[grid.idx(i, j, k)] +
              qq_rslt.by[grid.idx(i, j, k)] * qq_rslt.by[grid.idx(i, j, k)] +
              qq_rslt.bz[grid.idx(i, j, k)] * qq_rslt.bz[grid.idx(i, j, k)])) /
        qq_rslt.ro[grid.idx(i, j, k)];
  }
}
}  // namespace artificial_viscosity

template <typename Real> struct ArtificialViscosity {
  Config &config;
  Time<Real> &time;
  Grid<Real> &grid;
  GridDevice<Real> &grid_d;
  EOS<Real> &eos;
  MHD<Real> &mhd;
  MHDDevice<Real> &mhd_d;
  CudaKernelShape<Real> &cu_shape;

  Array3DDevice<Real> cc_d;
  Array3D<Real> cc;
  Real ep, fh;
  Real cs_fac, ca_fac, vv_fac;

  ArtificialViscosity(Model<Real> &model)
      : config(model.config), time(model.time), grid(model.grid_local),
        eos(model.eos), mhd(model.mhd), mhd_d(model.mhd_d), grid_d(model.grid_d),
        cu_shape(model.cu_shape), cc(grid.i_total, grid.j_total, grid.k_total),
        cc_d(grid.i_total, grid.j_total, grid.k_total) {
    this->ep = config.yaml_obj["artificial_viscosity"]["ep"].template as<Real>();
    this->fh = config.yaml_obj["artificial_viscosity"]["fh"].template as<Real>();
    this->cs_fac =
        config.yaml_obj["artificial_viscosity"]["cs_fac"].template as<Real>();
    this->ca_fac =
        config.yaml_obj["artificial_viscosity"]["ca_fac"].template as<Real>();
    this->vv_fac =
        config.yaml_obj["artificial_viscosity"]["vv_fac"].template as<Real>();
    assert(ep >= 0);
    assert(fh >= 0);
  }

  void characteristic_velocity_eval() {
    artificial_viscosity::characteristic_velocity_eval_kernel<Real>
        <<<cu_shape.grid_dim, cu_shape.block_dim>>>(
            cc_d, mhd_d.qq, grid_d, eos.gm, cs_fac, ca_fac, vv_fac);
  }

  void update(std::vector<Real> &dxyzi, Real *dxyzi_d, std::string direction) {
    int i0_ = 0;
    int i1_ = grid.i_total;
    int is = 0;
    int j0_ = 0;
    int j1_ = grid.j_total;
    int js = 0;
    int k0_ = 0;
    int k1_ = grid.k_total;
    int ks = 0;

    if (direction == "x") {
      i0_ = 2 * grid.is;
      i1_ = grid.i_total - 2 * grid.is;
      is = grid.is;
    } else if (direction == "y") {
      j0_ = 2 * grid.js;
      j1_ = grid.j_total - 2 * grid.js;
      js = grid.js;
    } else if (direction == "z") {
      k0_ = 2 * grid.ks;
      k1_ = grid.k_total - 2 * grid.ks;
      ks = grid.ks;
    }

    artificial_viscosity::update_ro_kernel<Real>
        <<<cu_shape.grid_dim, cu_shape.block_dim>>>(
            mhd_d.qq, mhd_d.qq_rslt, cc_d, grid_d, dxyzi_d, i0_, i1_, j0_, j1_,
            k0_, k1_, is, js, ks, ep, fh, time.dt);

    artificial_viscosity::update_vx_kernel<Real>
        <<<cu_shape.grid_dim, cu_shape.block_dim>>>(
            mhd_d.qq, mhd_d.qq_rslt, cc_d, grid_d, dxyzi_d, i0_, i1_, j0_, j1_,
            k0_, k1_, is, js, ks, ep, fh, time.dt);

    artificial_viscosity::update_vy_kernel<Real>
        <<<cu_shape.grid_dim, cu_shape.block_dim>>>(
            mhd_d.qq, mhd_d.qq_rslt, cc_d, grid_d, dxyzi_d, i0_, i1_, j0_, j1_,
            k0_, k1_, is, js, ks, ep, fh, time.dt);

    artificial_viscosity::update_vz_kernel<Real>
        <<<cu_shape.grid_dim, cu_shape.block_dim>>>(
            mhd_d.qq, mhd_d.qq_rslt, cc_d, grid_d, dxyzi_d, i0_, i1_, j0_, j1_,
            k0_, k1_, is, js, ks, ep, fh, time.dt);

    artificial_viscosity::update_bx_kernel<Real>
        <<<cu_shape.grid_dim, cu_shape.block_dim>>>(
            mhd_d.qq, mhd_d.qq_rslt, cc_d, grid_d, dxyzi_d, i0_, i1_, j0_, j1_,
            k0_, k1_, is, js, ks, ep, fh, time.dt);

    artificial_viscosity::update_by_kernel<Real>
        <<<cu_shape.grid_dim, cu_shape.block_dim>>>(
            mhd_d.qq, mhd_d.qq_rslt, cc_d, grid_d, dxyzi_d, i0_, i1_, j0_, j1_,
            k0_, k1_, is, js, ks, ep, fh, time.dt);

    artificial_viscosity::update_bz_kernel<Real>
        <<<cu_shape.grid_dim, cu_shape.block_dim>>>(
            mhd_d.qq, mhd_d.qq_rslt, cc_d, grid_d, dxyzi_d, i0_, i1_, j0_, j1_,
            k0_, k1_, is, js, ks, ep, fh, time.dt);

    artificial_viscosity::update_ph_kernel<Real>
        <<<cu_shape.grid_dim, cu_shape.block_dim>>>(
            mhd_d.qq, mhd_d.qq_rslt, cc_d, grid_d, dxyzi_d, i0_, i1_, j0_, j1_,
            k0_, k1_, is, js, ks, ep, fh, time.dt);

    artificial_viscosity::update_ei_kernel<Real>
        <<<cu_shape.grid_dim, cu_shape.block_dim>>>(
            mhd_d.qq, mhd_d.qq_rslt, cc_d, grid_d, dxyzi_d, i0_, i1_, j0_, j1_,
            k0_, k1_, is, js, ks, ep, fh, time.dt);
  }
};

}  // namespace mhd
}  // namespace miso
