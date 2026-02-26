#pragma once

#include "array3d.hpp"
#include "constants.hpp"
#include "cuda_util.cuh"
#include "grid.hpp"
#include "limiter.hpp"
#include "mhd_fields.hpp"

namespace miso {
namespace mhd {
namespace impl_cuda {
namespace artificial_viscosity {

using miso::limiter::dqq_eval;
using miso::limiter::flux_core;

template <typename Real, typename EOS>
__global__ void characteristic_velocity_eval_kernel(Array3DView<Real> cc_d,
                                                    FieldsView<const Real> qq,
                                                    GridView<Real> grid,
                                                    Real cs_fac, Real ca_fac,
                                                    Real vv_fac, EOS eos) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i <= grid.i_total - 1 && j <= grid.j_total - 1 && k <= grid.k_total - 1) {
    // clang-format off
    Real cs = eos.roeitocs(qq.ro(i, j, k), qq.ei(i, j, k));
    Real vv = util::sqrt(qq.vx(i, j, k) * qq.vx(i, j, k) +
                        qq.vy(i, j, k) * qq.vy(i, j, k) +
                        qq.vz(i, j, k) * qq.vz(i, j, k));
    Real ca = util::sqrt((qq.bx(i, j, k) * qq.bx(i, j, k) +
                         qq.by(i, j, k) * qq.by(i, j, k) +
                         qq.bz(i, j, k) * qq.bz(i, j, k)) /
                         qq.ro(i, j, k) * pii4<Real>);
    // clang-format on
    cc_d(i, j, k) = cs * cs_fac + vv * vv_fac + ca * ca_fac;
  }
}

template <typename Real>
__global__ void update_ro_kernel(FieldsView<Real> qq, FieldsView<Real> qq_rslt,
                                 Array3DView<Real> cc, GridView<Real> grid,
                                 Real *dxyzi, int i0_, int i1_, int j0_, int j1_,
                                 int k0_, int k1_, int is, int js, int ks,
                                 Real ep, Real fh, Real dt) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= i0_ && i < i1_ && j >= j0_ && j < j1_ && k >= k0_ && k < k1_) {
    // chracteristic velocity
    // clang-format off
    Real ccl = cc(i - is, j - js, k - ks);
    Real ccc = cc(i     , j     , k     );
    Real ccr = cc(i + is, j + js, k + ks);

    // density
    Real qql2 = qq.ro(i - 2 * is, j - 2 * js, k - 2 * ks);
    Real qql1 = qq.ro(i -     is, j -     js, k -     ks);
    Real qqc  = qq.ro(i         , j         , k         );
    Real qqr1 = qq.ro(i +     is, j +     js, k +     ks);
    Real qqr2 = qq.ro(i + 2 * is, j + 2 * js, k + 2 * ks);
    // clang-format on

    // dqq at i-is, j-js, k-2ks
    Real dqq_dw = dqq_eval(qql2, qql1, qqc, ep);
    // dqq at i, j, k
    Real dqq_cn = dqq_eval(qql1, qqc, qqr1, ep);
    // dqq at i+is, j+js, k+ks
    Real dqq_up = dqq_eval(qqc, qqr1, qqr2, ep);
    Real fro_dw =
        flux_core(qql1, qqc, dqq_dw, dqq_cn, Real(0.5) * (ccl + ccc), fh);
    Real fro_up =
        flux_core(qqc, qqr1, dqq_cn, dqq_up, Real(0.5) * (ccc + ccr), fh);

    qq_rslt.ro(i, j, k) =
        qq.ro(i, j, k) - (fro_up - fro_dw) * dxyzi[i * is + j * js + k * ks] * dt;
  }
}

template <typename Real>
__global__ void update_vx_kernel(FieldsView<Real> qq, FieldsView<Real> qq_rslt,
                                 Array3DView<Real> cc, GridView<Real> grid,
                                 Real *dxyzi, int i0_, int i1_, int j0_, int j1_,
                                 int k0_, int k1_, int is, int js, int ks,
                                 Real ep, Real fh, Real dt) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= i0_ && i < i1_ && j >= j0_ && j < j1_ && k >= k0_ && k < k1_) {
    // chracteristic velocity
    // clang-format off
    Real ccl = cc(i - is, j - js, k - ks);
    Real ccc = cc(i     , j     , k     );
    Real ccr = cc(i + is, j + js, k + ks);

    // x momentum
    Real qql2 = qq.ro(i - 2 * is, j - 2 * js, k - 2 * ks) *
                qq.vx(i - 2 * is, j - 2 * js, k - 2 * ks);
    Real qql1 = qq.ro(i -     is, j -     js, k -     ks) *
                qq.vx(i -     is, j -     js, k -     ks);
    Real qqc  = qq.ro(i         , j         , k         ) *
                qq.vx(i         , j         , k         );
    Real qqr1 = qq.ro(i +     is, j +     js, k +     ks) *
                qq.vx(i +     is, j +     js, k +     ks);
    Real qqr2 = qq.ro(i + 2 * is, j + 2 * js, k + 2 * ks) *
                qq.vx(i + 2 * is, j + 2 * js, k + 2 * ks);
    // clang-format on
    // dqq at i-is, j-js, k-2ks
    Real dqq_dw = dqq_eval(qql2, qql1, qqc, ep);
    // dqq at i, j, k
    Real dqq_cn = dqq_eval(qql1, qqc, qqr1, ep);
    // dqq at i+is, j+js, k+ks
    Real dqq_up = dqq_eval(qqc, qqr1, qqr2, ep);
    Real frx_dw =
        flux_core(qql1, qqc, dqq_dw, dqq_cn, Real(0.5) * (ccl + ccc), fh);
    Real frx_up =
        flux_core(qqc, qqr1, dqq_cn, dqq_up, Real(0.5) * (ccc + ccr), fh);

    qq_rslt.vx(i, j, k) =
        (qq.ro(i, j, k) * qq.vx(i, j, k) -
         (frx_up - frx_dw) * dxyzi[i * is + j * js + k * ks] * dt) /
        qq_rslt.ro(i, j, k);
  }
}

template <typename Real>
__global__ void update_vy_kernel(FieldsView<Real> qq, FieldsView<Real> qq_rslt,
                                 Array3DView<Real> cc, GridView<Real> grid,
                                 Real *dxyzi, int i0_, int i1_, int j0_, int j1_,
                                 int k0_, int k1_, int is, int js, int ks,
                                 Real ep, Real fh, Real dt) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= i0_ && i < i1_ && j >= j0_ && j < j1_ && k >= k0_ && k < k1_) {
    // chracteristic velocity
    // clang-format off
    Real ccl = cc(i - is, j - js, k - ks);
    Real ccc = cc(i     , j     , k     );
    Real ccr = cc(i + is, j + js, k + ks);

    // x momentum
    Real qql2 = qq.ro(i - 2 * is, j - 2 * js, k - 2 * ks) *
                qq.vy(i - 2 * is, j - 2 * js, k - 2 * ks);
    Real qql1 = qq.ro(i -     is, j -     js, k -     ks) *
                qq.vy(i -     is, j -     js, k -     ks);
    Real qqc  = qq.ro(i         , j         , k         ) *
                qq.vy(i         , j         , k         );
    Real qqr1 = qq.ro(i +     is, j +     js, k +     ks) *
                qq.vy(i +     is, j +     js, k +     ks);
    Real qqr2 = qq.ro(i + 2 * is, j + 2 * js, k + 2 * ks) *
                qq.vy(i + 2 * is, j + 2 * js, k + 2 * ks);
    // clang-format on
    // dqq at i-is, j-js, k-2ks
    Real dqq_dw = dqq_eval(qql2, qql1, qqc, ep);
    // dqq at i, j, k
    Real dqq_cn = dqq_eval(qql1, qqc, qqr1, ep);
    // dqq at i+is, j+js, k+ks
    Real dqq_up = dqq_eval(qqc, qqr1, qqr2, ep);
    Real fry_dw =
        flux_core(qql1, qqc, dqq_dw, dqq_cn, Real(0.5) * (ccl + ccc), fh);
    Real fry_up =
        flux_core(qqc, qqr1, dqq_cn, dqq_up, Real(0.5) * (ccc + ccr), fh);

    qq_rslt.vy(i, j, k) =
        (qq.ro(i, j, k) * qq.vy(i, j, k) -
         (fry_up - fry_dw) * dxyzi[i * is + j * js + k * ks] * dt) /
        qq_rslt.ro(i, j, k);
  }
}

template <typename Real>
__global__ void update_vz_kernel(FieldsView<Real> qq, FieldsView<Real> qq_rslt,
                                 Array3DView<Real> cc, GridView<Real> grid,
                                 Real *dxyzi, int i0_, int i1_, int j0_, int j1_,
                                 int k0_, int k1_, int is, int js, int ks,
                                 Real ep, Real fh, Real dt) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= i0_ && i < i1_ && j >= j0_ && j < j1_ && k >= k0_ && k < k1_) {
    // chracteristic velocity
    // clang-format off
    Real ccl = cc(i - is, j - js, k - ks);
    Real ccc = cc(i     , j     , k     );
    Real ccr = cc(i + is, j + js, k + ks);

    // x momentum
    Real qql2 = qq.ro(i - 2 * is, j - 2 * js, k - 2 * ks) *
                qq.vz(i - 2 * is, j - 2 * js, k - 2 * ks);
    Real qql1 = qq.ro(i -     is, j -     js, k -     ks) *
                qq.vz(i -     is, j -     js, k -     ks);
    Real qqc  = qq.ro(i         , j         , k         ) *
                qq.vz(i         , j         , k         );
    Real qqr1 = qq.ro(i +     is, j +     js, k +     ks) *
                qq.vz(i +     is, j +     js, k +     ks);
    Real qqr2 = qq.ro(i + 2 * is, j + 2 * js, k + 2 * ks) *
                qq.vz(i + 2 * is, j + 2 * js, k + 2 * ks);
    // clang-format on
    // dqq at i-is, j-js, k-2ks
    Real dqq_dw = dqq_eval(qql2, qql1, qqc, ep);
    // dqq at i, j, k
    Real dqq_cn = dqq_eval(qql1, qqc, qqr1, ep);
    // dqq at i+is, j+js, k+ks
    Real dqq_up = dqq_eval(qqc, qqr1, qqr2, ep);
    Real frz_dw =
        flux_core(qql1, qqc, dqq_dw, dqq_cn, Real(0.5) * (ccl + ccc), fh);
    Real frz_up =
        flux_core(qqc, qqr1, dqq_cn, dqq_up, Real(0.5) * (ccc + ccr), fh);

    qq_rslt.vz(i, j, k) =
        (qq.ro(i, j, k) * qq.vz(i, j, k) -
         (frz_up - frz_dw) * dxyzi[i * is + j * js + k * ks] * dt) /
        qq_rslt.ro(i, j, k);
  }
}

template <typename Real>
__global__ void update_bx_kernel(FieldsView<Real> qq, FieldsView<Real> qq_rslt,
                                 Array3DView<Real> cc, GridView<Real> grid,
                                 Real *dxyzi, int i0_, int i1_, int j0_, int j1_,
                                 int k0_, int k1_, int is, int js, int ks,
                                 Real ep, Real fh, Real dt) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= i0_ && i < i1_ && j >= j0_ && j < j1_ && k >= k0_ && k < k1_) {
    // chracteristic velocity
    // clang-format off
    Real ccl = cc(i - is, j - js, k - ks);
    Real ccc = cc(i     , j     , k     );
    Real ccr = cc(i + is, j + js, k + ks);

    Real qql2 = qq.bx(i - 2 * is, j - 2 * js, k - 2 * ks);
    Real qql1 = qq.bx(i -     is, j -     js, k -     ks);
    Real qqc  = qq.bx(i         , j         , k         );
    Real qqr1 = qq.bx(i +     is, j +     js, k +     ks);
    Real qqr2 = qq.bx(i + 2 * is, j + 2 * js, k + 2 * ks);
    // clang-format on
    // dqq at i-is, j-js, k-2ks
    Real dqq_dw = dqq_eval(qql2, qql1, qqc, ep);
    // dqq at i, j, k
    Real dqq_cn = dqq_eval(qql1, qqc, qqr1, ep);
    // dqq at i+is, j+js, k+ks
    Real dqq_up = dqq_eval(qqc, qqr1, qqr2, ep);
    Real fbx_dw =
        flux_core(qql1, qqc, dqq_dw, dqq_cn, Real(0.5) * (ccl + ccc), fh);
    Real fbx_up =
        flux_core(qqc, qqr1, dqq_cn, dqq_up, Real(0.5) * (ccc + ccr), fh);

    qq_rslt.bx(i, j, k) =
        qq.bx(i, j, k) - (fbx_up - fbx_dw) * dxyzi[i * is + j * js + k * ks] * dt;
  }
}

template <typename Real>
__global__ void update_by_kernel(FieldsView<Real> qq, FieldsView<Real> qq_rslt,
                                 Array3DView<Real> cc, GridView<Real> grid,
                                 Real *dxyzi, int i0_, int i1_, int j0_, int j1_,
                                 int k0_, int k1_, int is, int js, int ks,
                                 Real ep, Real fh, Real dt) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= i0_ && i < i1_ && j >= j0_ && j < j1_ && k >= k0_ && k < k1_) {
    // chracteristic velocity
    // clang-format off
    Real ccl = cc(i - is, j - js, k - ks);
    Real ccc = cc(i     , j     , k     );
    Real ccr = cc(i + is, j + js, k + ks);

    Real qql2 = qq.by(i - 2 * is, j - 2 * js, k - 2 * ks);
    Real qql1 = qq.by(i -     is, j -     js, k -     ks);
    Real qqc  = qq.by(i         , j         , k         );
    Real qqr1 = qq.by(i +     is, j +     js, k +     ks);
    Real qqr2 = qq.by(i + 2 * is, j + 2 * js, k + 2 * ks);
    // clang-format on

    // dqq at i-is, j-js, k-2ks
    Real dqq_dw = dqq_eval(qql2, qql1, qqc, ep);
    // dqq at i, j, k
    Real dqq_cn = dqq_eval(qql1, qqc, qqr1, ep);
    // dqq at i+is, j+js, k+ks
    Real dqq_up = dqq_eval(qqc, qqr1, qqr2, ep);
    Real fby_dw =
        flux_core(qql1, qqc, dqq_dw, dqq_cn, Real(0.5) * (ccl + ccc), fh);
    Real fby_up =
        flux_core(qqc, qqr1, dqq_cn, dqq_up, Real(0.5) * (ccc + ccr), fh);

    qq_rslt.by(i, j, k) =
        qq.by(i, j, k) - (fby_up - fby_dw) * dxyzi[i * is + j * js + k * ks] * dt;
  }
}

template <typename Real>
__global__ void update_bz_kernel(FieldsView<Real> qq, FieldsView<Real> qq_rslt,
                                 Array3DView<Real> cc, GridView<Real> grid,
                                 Real *dxyzi, int i0_, int i1_, int j0_, int j1_,
                                 int k0_, int k1_, int is, int js, int ks,
                                 Real ep, Real fh, Real dt) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= i0_ && i < i1_ && j >= j0_ && j < j1_ && k >= k0_ && k < k1_) {
    // chracteristic velocity
    // clang-format off
    Real ccl = cc(i - is, j - js, k - ks);
    Real ccc = cc(i     , j     , k     );
    Real ccr = cc(i + is, j + js, k + ks);

    Real qql2 = qq.bz(i - 2 * is, j - 2 * js, k - 2 * ks);
    Real qql1 = qq.bz(i -     is, j -     js, k -     ks);
    Real qqc  = qq.bz(i         , j         , k         );
    Real qqr1 = qq.bz(i +     is, j +     js, k +     ks);
    Real qqr2 = qq.bz(i + 2 * is, j + 2 * js, k + 2 * ks);
    // clang-format on

    // dqq at i-is, j-js, k-2ks
    Real dqq_dw = dqq_eval(qql2, qql1, qqc, ep);
    // dqq at i, j, k
    Real dqq_cn = dqq_eval(qql1, qqc, qqr1, ep);
    // dqq at i+is, j+js, k+ks
    Real dqq_up = dqq_eval(qqc, qqr1, qqr2, ep);
    Real fbz_dw =
        flux_core(qql1, qqc, dqq_dw, dqq_cn, Real(0.5) * (ccl + ccc), fh);
    Real fbz_up =
        flux_core(qqc, qqr1, dqq_cn, dqq_up, Real(0.5) * (ccc + ccr), fh);

    qq_rslt.bz(i, j, k) =
        qq.bz(i, j, k) - (fbz_up - fbz_dw) * dxyzi[i * is + j * js + k * ks] * dt;
  }
}

template <typename Real>
__global__ void update_ph_kernel(FieldsView<Real> qq, FieldsView<Real> qq_rslt,
                                 Array3DView<Real> cc, GridView<Real> grid,
                                 Real *dxyzi, int i0_, int i1_, int j0_, int j1_,
                                 int k0_, int k1_, int is, int js, int ks,
                                 Real ep, Real fh, Real dt) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= i0_ && i < i1_ && j >= j0_ && j < j1_ && k >= k0_ && k < k1_) {
    // chracteristic velocity
    // clang-format off
    Real ccl = cc(i - is, j - js, k - ks);
    Real ccc = cc(i     , j     , k     );
    Real ccr = cc(i + is, j + js, k + ks);

    Real qql2 = qq.ph(i - 2 * is, j - 2 * js, k - 2 * ks);
    Real qql1 = qq.ph(i -     is, j -     js, k -     ks);
    Real qqc  = qq.ph(i         , j         , k         );
    Real qqr1 = qq.ph(i +     is, j +     js, k +     ks);
    Real qqr2 = qq.ph(i + 2 * is, j + 2 * js, k + 2 * ks);
    // clang-format on
    // dqq at i-is, j-js, k-2ks
    Real dqq_dw = dqq_eval(qql2, qql1, qqc, ep);
    // dqq at i, j, k
    Real dqq_cn = dqq_eval(qql1, qqc, qqr1, ep);
    // dqq at i+is, j+js, k+ks
    Real dqq_up = dqq_eval(qqc, qqr1, qqr2, ep);
    Real fph_dw =
        flux_core(qql1, qqc, dqq_dw, dqq_cn, Real(0.5) * (ccl + ccc), fh);
    Real fph_up =
        flux_core(qqc, qqr1, dqq_cn, dqq_up, Real(0.5) * (ccc + ccr), fh);

    qq_rslt.ph(i, j, k) =
        qq.ph(i, j, k) - (fph_up - fph_dw) * dxyzi[i * is + j * js + k * ks] * dt;
  }
}

template <typename Real>
__global__ void update_ei_kernel(FieldsView<Real> qq, FieldsView<Real> qq_rslt,
                                 Array3DView<Real> cc, GridView<Real> grid,
                                 Real *dxyzi, int i0_, int i1_, int j0_, int j1_,
                                 int k0_, int k1_, int is, int js, int ks,
                                 Real ep, Real fh, Real dt) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= i0_ && i < i1_ && j >= j0_ && j < j1_ && k >= k0_ && k < k1_) {
    // chracteristic velocity
    // clang-format off
    Real ccl = cc(i - is, j - js, k - ks);
    Real ccc = cc(i     , j     , k     );
    Real ccr = cc(i + is, j + js, k + ks);

    // total energy
    Real qql2 = qq.ro(i - 2 * is, j - 2 * js, k - 2 * ks) *
                qq.ei(i - 2 * is, j - 2 * js, k - 2 * ks);
    Real qql1 = qq.ro(i -     is, j -     js, k -     ks) *
                qq.ei(i -     is, j -     js, k -     ks);
    Real qqc  = qq.ro(i         , j         , k         ) *
                qq.ei(i         , j         , k         );
    Real qqr1 = qq.ro(i +     is, j +     js, k +     ks) *
                qq.ei(i +     is, j +     js, k +     ks);
    Real qqr2 = qq.ro(i + 2 * is, j + 2 * js, k + 2 * ks) *
                qq.ei(i + 2 * is, j + 2 * js, k + 2 * ks);
    // clang-format on
    // dqq at i-is, j-js, k-2ks
    Real dqq_dw = dqq_eval(qql2, qql1, qqc, ep);
    // dqq at i, j, k
    Real dqq_cn = dqq_eval(qql1, qqc, qqr1, ep);
    // dqq at i+is, j+js, k+ks
    Real dqq_up = dqq_eval(qqc, qqr1, qqr2, ep);
    Real fei_dw =
        flux_core(qql1, qqc, dqq_dw, dqq_cn, Real(0.5) * (ccl + ccc), fh);
    Real fei_up =
        flux_core(qqc, qqr1, dqq_cn, dqq_up, Real(0.5) * (ccc + ccr), fh);

    // Et: Total energy per unit volume, note that ei is internal energy per unit mass
    Real Et =
        qq.ro(i, j, k) * qq.ei(i, j, k) +
        0.5 * qq.ro(i, j, k) *
            (qq.vx(i, j, k) * qq.vx(i, j, k) + qq.vy(i, j, k) * qq.vy(i, j, k) +
             qq.vz(i, j, k) * qq.vz(i, j, k)) +
        pii8<Real> *
            (qq.bx(i, j, k) * qq.bx(i, j, k) + qq.by(i, j, k) * qq.by(i, j, k) +
             qq.bz(i, j, k) * qq.bz(i, j, k));

    qq_rslt.ei(i, j, k) =
        (Et - (fei_up - fei_dw) * dxyzi[i * is + j * js + k * ks] * dt -
         0.5 * qq_rslt.ro(i, j, k) *
             (qq_rslt.vx(i, j, k) * qq_rslt.vx(i, j, k) +
              qq_rslt.vy(i, j, k) * qq_rslt.vy(i, j, k) +
              qq_rslt.vz(i, j, k) * qq_rslt.vz(i, j, k)) -
         pii8<Real> * (qq_rslt.bx(i, j, k) * qq_rslt.bx(i, j, k) +
                       qq_rslt.by(i, j, k) * qq_rslt.by(i, j, k) +
                       qq_rslt.bz(i, j, k) * qq_rslt.bz(i, j, k))) /
        qq_rslt.ro(i, j, k);
  }
}

}  // namespace artificial_viscosity

template <typename Real> struct ArtificialViscosity {
  Grid<Real, backend::CUDA> &grid;
  cuda::KernelShape3D &cu_shape;

  /// @brief Characteristic velocity cs_fac*cs + ca_fac*ca + vv_fac*vv
  Array3D<Real, backend::CUDA> cc;
  /// @brief Parameters for generalized minmod limiter
  Real ep;
  /// @brief Parameters for amplitude of artificial viscosity flux
  Real fh;
  /// @brief Characteristic velocity factor for sound speed
  Real cs_fac;
  /// @brief Characteristic velocity factor for Alfvén speed
  Real ca_fac;
  /// @brief Characteristic velocity factor for fluid velocity
  Real vv_fac;

  /// @brief Constructor for ArtificialViscosity
  ArtificialViscosity(Config &config, Grid<Real, backend::CUDA> &grid,
                      cuda::KernelShape3D &cu_shape)
      : grid(grid), cu_shape(cu_shape),
        cc(grid.i_total, grid.j_total, grid.k_total) {
    ep = config["mhd"]["artificial_viscosity"]["ep"].as<Real>();
    fh = config["mhd"]["artificial_viscosity"]["fh"].as<Real>();
    cs_fac = config["mhd"]["artificial_viscosity"]["cs_fac"].as<Real>();
    ca_fac = config["mhd"]["artificial_viscosity"]["ca_fac"].as<Real>();
    vv_fac = config["mhd"]["artificial_viscosity"]["vv_fac"].as<Real>();
    assert(ep >= 0);
    assert(fh >= 0);
  }

  /// @brief Evaluate the characteristic velocity
  template <typename EOS>
  void characteristic_velocity_eval(const Fields<Real, backend::CUDA> &qq,
                                    const EOS &eos) {
    artificial_viscosity::characteristic_velocity_eval_kernel<Real, EOS>
        <<<cu_shape.grid_dim, cu_shape.block_dim>>>(
            cc.view(), qq.view(), grid.view(), cs_fac, ca_fac, vv_fac, eos);
  }

  void update(Fields<Real, backend::CUDA> &qq,
              Fields<Real, backend::CUDA> &qq_rslt, Direction direction,
              const Real dt) {
    int i0_ = 0;
    int i1_ = grid.i_total;
    int is = 0;
    int j0_ = 0;
    int j1_ = grid.j_total;
    int js = 0;
    int k0_ = 0;
    int k1_ = grid.k_total;
    int ks = 0;
    Real *dxyzi = nullptr;
    if (direction == Direction::X) {
      i0_ = 2 * grid.is;
      i1_ = grid.i_total - 2 * grid.is;
      is = grid.is;
      dxyzi = grid.dxi;
    } else if (direction == Direction::Y) {
      j0_ = 2 * grid.js;
      j1_ = grid.j_total - 2 * grid.js;
      js = grid.js;
      dxyzi = grid.dyi;
    } else if (direction == Direction::Z) {
      k0_ = 2 * grid.ks;
      k1_ = grid.k_total - 2 * grid.ks;
      ks = grid.ks;
      dxyzi = grid.dzi;
    }

    artificial_viscosity::update_ro_kernel<Real>
        <<<cu_shape.grid_dim, cu_shape.block_dim>>>(
            qq.view(), qq_rslt.view(), cc.view(), grid.view(), dxyzi, i0_, i1_,
            j0_, j1_, k0_, k1_, is, js, ks, ep, fh, dt);

    artificial_viscosity::update_vx_kernel<Real>
        <<<cu_shape.grid_dim, cu_shape.block_dim>>>(
            qq.view(), qq_rslt.view(), cc.view(), grid.view(), dxyzi, i0_, i1_,
            j0_, j1_, k0_, k1_, is, js, ks, ep, fh, dt);

    artificial_viscosity::update_vy_kernel<Real>
        <<<cu_shape.grid_dim, cu_shape.block_dim>>>(
            qq.view(), qq_rslt.view(), cc.view(), grid.view(), dxyzi, i0_, i1_,
            j0_, j1_, k0_, k1_, is, js, ks, ep, fh, dt);

    artificial_viscosity::update_vz_kernel<Real>
        <<<cu_shape.grid_dim, cu_shape.block_dim>>>(
            qq.view(), qq_rslt.view(), cc.view(), grid.view(), dxyzi, i0_, i1_,
            j0_, j1_, k0_, k1_, is, js, ks, ep, fh, dt);

    artificial_viscosity::update_bx_kernel<Real>
        <<<cu_shape.grid_dim, cu_shape.block_dim>>>(
            qq.view(), qq_rslt.view(), cc.view(), grid.view(), dxyzi, i0_, i1_,
            j0_, j1_, k0_, k1_, is, js, ks, ep, fh, dt);

    artificial_viscosity::update_by_kernel<Real>
        <<<cu_shape.grid_dim, cu_shape.block_dim>>>(
            qq.view(), qq_rslt.view(), cc.view(), grid.view(), dxyzi, i0_, i1_,
            j0_, j1_, k0_, k1_, is, js, ks, ep, fh, dt);

    artificial_viscosity::update_bz_kernel<Real>
        <<<cu_shape.grid_dim, cu_shape.block_dim>>>(
            qq.view(), qq_rslt.view(), cc.view(), grid.view(), dxyzi, i0_, i1_,
            j0_, j1_, k0_, k1_, is, js, ks, ep, fh, dt);

    artificial_viscosity::update_ph_kernel<Real>
        <<<cu_shape.grid_dim, cu_shape.block_dim>>>(
            qq.view(), qq_rslt.view(), cc.view(), grid.view(), dxyzi, i0_, i1_,
            j0_, j1_, k0_, k1_, is, js, ks, ep, fh, dt);

    artificial_viscosity::update_ei_kernel<Real>
        <<<cu_shape.grid_dim, cu_shape.block_dim>>>(
            qq.view(), qq_rslt.view(), cc.view(), grid.view(), dxyzi, i0_, i1_,
            j0_, j1_, k0_, k1_, is, js, ks, ep, fh, dt);
  }
};

}  // namespace impl_cuda
}  // namespace mhd
}  // namespace miso
