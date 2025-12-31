#pragma once

#include <miso/array3d.hpp>
#include <miso/constants.hpp>
#include <miso/cuda_compat.hpp>
#include <miso/cuda_util.cuh>
#include <miso/env.hpp>
#include <miso/mhd_artificial_viscosity_gpu.cuh>
#include <miso/mhd_fields.hpp>
#include <miso/mhd_halo_exchange.hpp>

namespace miso {
namespace mhd {
namespace impl_cuda {

template <typename Real> struct TimeStep {
  Real *min_values_device = nullptr;
  Real *min_values_host = nullptr;
  size_t shared_mem_size = 0;
  int n_blocks;

  TimeStep(cuda::KernelShape3D &cu_shape)
      : n_blocks(cu_shape.grid_dim.x * cu_shape.grid_dim.y *
                 cu_shape.grid_dim.z) {
    min_values_host = new Real[n_blocks];
    shared_mem_size = sizeof(Real) * cu_shape.block_dim.x * cu_shape.block_dim.y *
                      cu_shape.block_dim.z;
    MISO_CUDA_CHECK(cudaMalloc(&min_values_device, sizeof(Real) * n_blocks));
  }

  ~TimeStep() {
    delete[] min_values_host;
    MISO_CUDA_CHECK(cudaFree(min_values_device));
  }

  void copy_to_host() const {
    MISO_CUDA_CHECK(cudaMemcpy(min_values_host, min_values_device,
                               sizeof(Real) * n_blocks, cudaMemcpyDeviceToHost));
  }
};

template <typename Real>
__global__ void cfl_kernel(FieldsView<Real> qq, GridView<Real> grid,
                           Real *dt_mins, Real cfl_number, Real eos_gm) {
  extern __shared__ Real dt_min_shared_in_block[];

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  Real dt = 1.e10;
  if (i >= grid.i_margin && i < grid.i_total - grid.i_margin &&
      j >= grid.j_margin && j < grid.j_total - grid.j_margin &&
      k >= grid.k_margin && k < grid.k_total - grid.k_margin) {
    // clang-format off
    Real cs = std::sqrt(eos_gm * (eos_gm - 1.0) * qq.ei(i, j, k));
    Real vv = std::sqrt( + qq.vx(i, j, k) * qq.vx(i, j, k)
                         + qq.vy(i, j, k) * qq.vy(i, j, k)
                         + qq.vz(i, j, k) * qq.vz(i, j, k));
    Real ca = std::sqrt((+ qq.bx(i, j, k) * qq.bx(i, j, k)
                         + qq.by(i, j, k) * qq.by(i, j, k)
                         + qq.bz(i, j, k) * qq.bz(i, j, k)) /
                        qq.ro(i, j, k) * pii4<Real>);
    Real total_vel = (cs + vv + ca);
    // clang-format on
    dt = cfl_number * util::min3(grid.dx[i], grid.dy[j], grid.dz[k]) / total_vel;
  }

  int thread_id = threadIdx.z * blockDim.y * blockDim.x +
                  threadIdx.y * blockDim.x + threadIdx.x;
  dt_min_shared_in_block[thread_id] = dt;
  __syncthreads();

  for (int s = blockDim.x * blockDim.y * blockDim.z / 2; s > 0; s >>= 1) {
    if (thread_id < s) {
      dt_min_shared_in_block[thread_id] =
          util::fmin_safe(dt_min_shared_in_block[thread_id],
                          dt_min_shared_in_block[thread_id + s]);
    }
    __syncthreads();
  }

  // ブロックのスレッド0が結果を書き出し
  if (thread_id == 0) {
    int block_id =
        blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x;
    dt_mins[block_id] = dt_min_shared_in_block[0];
  }
}

template <typename Real>
__device__ inline Real space_centered_4th(const Array3DView<Real> qq1, Real dxyzi,
                                          int i, int j, int k, int is, int js,
                                          int ks, const GridView<Real> &grid) {
  // clang-format off
  return (     -qq1(i + 2 * is, j + 2 * js, k + 2 * ks) +
          8.0 * qq1(i +     is, j +     js, k +     ks) -
          8.0 * qq1(i -     is, j -     js, k -     ks) +
                qq1(i - 2 * is, j - 2 * js, k - 2 * ks)) *
         inv12<Real> * dxyzi;
};
// clang-format on

template <typename Real>
__device__ inline Real space_centered_4th(const Array3DView<Real> qq1,
                                          const Array3DView<Real> qq2, Real dxyzi,
                                          int i, int j, int k, int is, int js,
                                          int ks, const GridView<Real> &grid) {

  // clang-format off
  return (    - qq1(i + 2 * is, j + 2 * js, k + 2 * ks) *
                qq2(i + 2 * is, j + 2 * js, k + 2 * ks) +
          8.0 * qq1(i +     is, j +     js, k +     ks) *
                qq2(i +     is, j +     js, k +     ks) -
          8.0 * qq1(i -     is, j -     js, k -     ks) *
                qq2(i -     is, j -     js, k -     ks) +
                qq1(i - 2 * is, j - 2 * js, k - 2 * ks) *
                qq2(i - 2 * is, j - 2 * js, k - 2 * ks)) *
         inv12<Real> * dxyzi;
};
// clang-format on

template <typename Real>
__device__ inline Real
space_centered_4th(const Array3DView<Real> qq1, const Array3DView<Real> qq2,
                   const Array3DView<Real> qq3, Real dxyzi, int i, int j, int k,
                   int is, int js, int ks, const GridView<Real> &grid) {
  // clang-format off
  return (    - qq1(i + 2 * is, j + 2 * js, k + 2 * ks) *
                qq2(i + 2 * is, j + 2 * js, k + 2 * ks) *
                qq3(i + 2 * is, j + 2 * js, k + 2 * ks) +
          8.0 * qq1(i +     is, j +     js, k +     ks) *
                qq2(i +     is, j +     js, k +     ks) *
                qq3(i +     is, j +     js, k +     ks) -
          8.0 * qq1(i -     is, j -     js, k -     ks) *
                qq2(i -     is, j -     js, k -     ks) *
                qq3(i -     is, j -     js, k -     ks) +
                qq1(i - 2 * is, j - 2 * js, k - 2 * ks) *
                qq2(i - 2 * is, j - 2 * js, k - 2 * ks) *
                qq3(i - 2 * is, j - 2 * js, k - 2 * ks)) *
         inv12<Real> * dxyzi;
};
// clang-format on

template <typename Real>
__global__ void pr_bb_ht_vb_kernel(FieldsView<Real> qq_argm, Array3DView<Real> pr,
                                   Array3DView<Real> bb, Array3DView<Real> ht,
                                   Array3DView<Real> vb, GridView<Real> grid,
                                   Real eos_gm) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < 0 || i >= grid.i_total || j < 0 || j >= grid.j_total || k < 0 ||
      k >= grid.k_total)
    return;

  // gas pressure
  pr(i, j, k) = qq_argm.ro(i, j, k) * qq_argm.ei(i, j, k) * (eos_gm - 1.0);
  // squared magnetic strength
  bb(i, j, k) = qq_argm.bx(i, j, k) * qq_argm.bx(i, j, k) +
                qq_argm.by(i, j, k) * qq_argm.by(i, j, k) +
                qq_argm.bz(i, j, k) * qq_argm.bz(i, j, k);

  // // enthalpy + 2*magnetic energy + kinetic energy
  // clang-format off
  ht(i, j, k) =
      +qq_argm.ro(i, j, k) * qq_argm.ei(i, j, k) +
      pr(i, j, k) + bb(i, j, k) * pii4<Real> +
      0.5 * qq_argm.ro(i, j, k) *
          (+ qq_argm.vx(i, j, k) * qq_argm.vx(i, j, k)
           + qq_argm.vy(i, j, k) * qq_argm.vy(i, j, k)
           + qq_argm.vz(i, j, k) * qq_argm.vz(i, j, k));
  // v dot b
  vb(i, j, k) =
      + qq_argm.vx(i, j, k) * qq_argm.bx(i, j, k)
      + qq_argm.vy(i, j, k) * qq_argm.by(i, j, k)
      + qq_argm.vz(i, j, k) * qq_argm.bz(i, j, k);
  // clang-format on
}

template <typename Real>
__device__ inline bool compute_index_within_margin(int &i, int &j, int &k,
                                                   const GridView<Real> &grid) {
  i = blockIdx.x * blockDim.x + threadIdx.x + grid.i_margin;
  j = blockIdx.y * blockDim.y + threadIdx.y + grid.j_margin;
  k = blockIdx.z * blockDim.z + threadIdx.z + grid.k_margin;

  return !(i < grid.i_margin || i >= grid.i_total - grid.i_margin ||
           j < grid.j_margin || j >= grid.j_total - grid.j_margin ||
           k < grid.k_margin || k >= grid.k_total - grid.k_margin);
}

template <typename Real>
__global__ void
update_ro_kernel(FieldsView<Real> qq_orgn, FieldsView<Real> qq_argm,
                 FieldsView<Real> qq_rslt, GridView<Real> grid, Real dt) {
  int i, j, k;
  if (!compute_index_within_margin(i, j, k, grid))
    return;

  // equation of continuity
  qq_rslt.ro(i, j, k) =
      qq_orgn.ro(i, j, k) +
      dt * (-space_centered_4th(qq_argm.ro, qq_argm.vx, grid.dxi[i], i, j, k,
                                grid.is, 0, 0, grid) -
            space_centered_4th(qq_argm.ro, qq_argm.vy, grid.dyi[j], i, j, k, 0,
                               grid.js, 0, grid) -
            space_centered_4th(qq_argm.ro, qq_argm.vz, grid.dzi[k], i, j, k, 0, 0,
                               grid.ks, grid));
}

template <typename Real, typename Source>
__global__ void update_vx_kernel(FieldsView<Real> qq_orgn,
                                 FieldsView<Real> qq_argm,
                                 FieldsView<Real> qq_rslt, Array3DView<Real> pr,
                                 Array3DView<Real> bb, Source source,
                                 GridView<Real> grid, Real dt) {
  int i, j, k;
  if (!compute_index_within_margin(i, j, k, grid))
    return;

  // clang-format off
  // x equation of motion
  qq_rslt.vx(i, j, k) =
      (qq_orgn.ro(i, j, k) * qq_orgn.vx(i, j, k) +
       dt *
           (- space_centered_4th(qq_argm.ro, qq_argm.vx, qq_argm.vx, grid.dxi[i], i, j, k, grid.is, 0, 0, grid)
            - space_centered_4th(qq_argm.ro, qq_argm.vx, qq_argm.vy, grid.dyi[j], i, j, k, 0, grid.js, 0, grid)
            - space_centered_4th(qq_argm.ro, qq_argm.vx, qq_argm.vz, grid.dzi[k], i, j, k, 0, 0, grid.ks, grid)
            - space_centered_4th(pr, grid.dxi[i], i, j, k, grid.is, 0, 0, grid)
            - space_centered_4th(bb, grid.dxi[i], i, j, k, grid.is, 0, 0, grid) * pii8<Real>
            + pii4<Real> * (+ space_centered_4th(qq_argm.bx, qq_argm.bx, grid.dxi[i], i, j, k, grid.is, 0, 0, grid)
                            + space_centered_4th(qq_argm.bx, qq_argm.by, grid.dyi[j], i, j, k, 0, grid.js, 0, grid)
                            + space_centered_4th(qq_argm.bx, qq_argm.bz, grid.dzi[k], i, j, k, 0, 0, grid.ks, grid))
            + source.vx(qq_argm, i, j, k)
            )) / qq_rslt.ro(i, j, k);
  // clang-format on
}

template <typename Real, typename Source>
__global__ void update_vy_kernel(FieldsView<Real> qq_orgn,
                                 FieldsView<Real> qq_argm,
                                 FieldsView<Real> qq_rslt, Array3DView<Real> pr,
                                 Array3DView<Real> bb, Source source,
                                 GridView<Real> grid, Real dt) {
  int i, j, k;
  if (!compute_index_within_margin(i, j, k, grid))
    return;

  // y equation of motion
  qq_rslt.vy(i, j, k) =
      (qq_orgn.ro(i, j, k) * qq_orgn.vy(i, j, k) +
       // clang-format off
       dt *
           (- space_centered_4th(qq_argm.ro, qq_argm.vy, qq_argm.vx, grid.dxi[i], i, j, k, grid.is, 0, 0, grid)
            - space_centered_4th(qq_argm.ro, qq_argm.vy, qq_argm.vy, grid.dyi[j], i, j, k, 0, grid.js, 0, grid)
            - space_centered_4th(qq_argm.ro, qq_argm.vy, qq_argm.vz, grid.dzi[k], i, j, k, 0, 0, grid.ks, grid)
            - space_centered_4th(pr, grid.dyi[j], i, j, k, 0, grid.js, 0, grid)
            - space_centered_4th(bb, grid.dyi[j], i, j, k, 0, grid.js, 0, grid) * pii8<Real>
            + pii4<Real> * (+ space_centered_4th(qq_argm.by, qq_argm.bx, grid.dxi[i], i, j, k, grid.is, 0, 0, grid)
                            + space_centered_4th(qq_argm.by, qq_argm.by, grid.dyi[j], i, j, k, 0, grid.js, 0, grid)
                            + space_centered_4th(qq_argm.by, qq_argm.bz, grid.dzi[k], i, j, k, 0, 0, grid.ks, grid))
            + source.vy(qq_argm, i, j, k)
            )) / qq_rslt.ro(i, j, k);
  // clang-format on
}

template <typename Real, typename Source>
__global__ void update_vz_kernel(FieldsView<Real> qq_orgn,
                                 FieldsView<Real> qq_argm,
                                 FieldsView<Real> qq_rslt, Array3DView<Real> pr,
                                 Array3DView<Real> bb, Source source,
                                 GridView<Real> grid, Real dt) {
  int i, j, k;
  if (!compute_index_within_margin(i, j, k, grid))
    return;

  // z equation of motion
  // clang-format off
  qq_rslt.vz(i, j, k) =
      (qq_orgn.ro(i, j, k) * qq_orgn.vz(i, j, k) +
       dt *
           ( - space_centered_4th(qq_argm.ro, qq_argm.vz, qq_argm.vx, grid.dxi[i], i, j, k, grid.is, 0, 0, grid)
             - space_centered_4th(qq_argm.ro, qq_argm.vz, qq_argm.vy, grid.dyi[j], i, j, k, 0, grid.js, 0, grid)
             - space_centered_4th(qq_argm.ro, qq_argm.vz, qq_argm.vz, grid.dzi[k], i, j, k, 0, 0, grid.ks, grid)
             - space_centered_4th(pr, grid.dzi[k], i, j, k, 0, 0, grid.ks, grid)
             - space_centered_4th(bb, grid.dzi[k], i, j, k, 0, 0, grid.ks, grid) * pii8<Real>
             + pii4<Real> * (+ space_centered_4th(qq_argm.bz, qq_argm.bx, grid.dxi[i], i, j, k, grid.is, 0, 0, grid)
                             + space_centered_4th(qq_argm.bz, qq_argm.by, grid.dyi[j], i, j, k, 0, grid.js, 0, grid)
                             + space_centered_4th(qq_argm.bz, qq_argm.bz, grid.dzi[k], i, j, k, 0, 0, grid.ks, grid))
             + source.vz(qq_argm, i, j, k)
          )) / qq_rslt.ro(i, j, k);
  // clang-format on
}

template <typename Real>
__global__ void
update_bx_kernel(FieldsView<Real> qq_orgn, FieldsView<Real> qq_argm,
                 FieldsView<Real> qq_rslt, GridView<Real> grid, Real dt) {
  int i, j, k;
  if (!compute_index_within_margin(i, j, k, grid))
    return;

  // x magnetic induction
  // clang-format off
  qq_rslt.bx(i, j, k) =
      qq_orgn.bx(i, j, k) +
      dt * (- space_centered_4th(qq_argm.vy, qq_argm.bx, grid.dyi[j], i, j, k, 0, grid.js, 0, grid)
            - space_centered_4th(qq_argm.vz, qq_argm.bx, grid.dzi[k], i, j, k, 0, 0, grid.ks, grid)
            + space_centered_4th(qq_argm.vx, qq_argm.by, grid.dyi[j], i, j, k, 0, grid.js, 0, grid)
            + space_centered_4th(qq_argm.vx, qq_argm.bz, grid.dzi[k], i, j, k, 0, 0, grid.ks, grid)
            - space_centered_4th(qq_argm.ph, grid.dxi[i], i, j, k, grid.is, 0, 0, grid));
  // clang-format on
}

template <typename Real>
__global__ void
update_by_kernel(FieldsView<Real> qq_orgn, FieldsView<Real> qq_argm,
                 FieldsView<Real> qq_rslt, GridView<Real> grid, Real dt) {
  int i, j, k;
  if (!compute_index_within_margin(i, j, k, grid))
    return;

  // y magnetic induction
  // clang-format off
  qq_rslt.by(i, j, k) =
      qq_orgn.by(i, j, k) +
      dt * (- space_centered_4th(qq_argm.vx, qq_argm.by, grid.dxi[i], i, j, k, grid.is, 0, 0, grid)
            - space_centered_4th(qq_argm.vz, qq_argm.by, grid.dzi[k], i, j, k, 0, 0, grid.ks, grid)
            + space_centered_4th(qq_argm.vy, qq_argm.bx, grid.dxi[i], i, j, k, grid.is, 0, 0, grid)
            + space_centered_4th(qq_argm.vy, qq_argm.bz, grid.dzi[k], i, j, k, 0, 0, grid.ks, grid)
            - space_centered_4th(qq_argm.ph, grid.dyi[j], i, j, k, 0, grid.js, 0, grid));
  // clang-format on
}

template <typename Real>
__global__ void
update_bz_kernel(FieldsView<Real> qq_orgn, FieldsView<Real> qq_argm,
                 FieldsView<Real> qq_rslt, GridView<Real> grid, Real dt) {
  int i, j, k;
  if (!compute_index_within_margin(i, j, k, grid))
    return;

  // z magnetic induction
  // clang-format off
  qq_rslt.bz(i, j, k) =
      qq_orgn.bz(i, j, k) +
      dt * (- space_centered_4th(qq_argm.vx, qq_argm.bz, grid.dxi[i], i, j, k, grid.is, 0, 0, grid)
            - space_centered_4th(qq_argm.vy, qq_argm.bz, grid.dyi[j], i, j, k, 0, grid.js, 0, grid)
            + space_centered_4th(qq_argm.vz, qq_argm.bx, grid.dxi[i], i, j, k, grid.is, 0, 0, grid)
            + space_centered_4th(qq_argm.vz, qq_argm.by, grid.dyi[j], i, j, k, 0, grid.js, 0, grid)
            - space_centered_4th(qq_argm.ph, grid.dzi[k], i, j, k, 0, 0, grid.ks, grid));
  // clang-format on
}

template <typename Real>
__global__ void update_ph_kernel(FieldsView<Real> qq_orgn,
                                 FieldsView<Real> qq_argm,
                                 FieldsView<Real> qq_rslt, Real ch_divb_square,
                                 Real tau_divb, GridView<Real> grid, Real dt) {
  int i, j, k;
  if (!compute_index_within_margin(i, j, k, grid))
    return;

  // div B factor
  // clang-format off
  qq_rslt.ph(i, j, k) =
      (qq_orgn.ph(i, j, k) +
       dt * ch_divb_square *
           (- space_centered_4th(qq_argm.bx, grid.dxi[i], i, j, k, grid.is, 0, 0, grid)
            - space_centered_4th(qq_argm.by, grid.dyi[j], i, j, k, 0, grid.js, 0, grid)
            - space_centered_4th(qq_argm.bz, grid.dzi[k], i, j, k, 0, 0, grid.ks, grid))) *
      std::exp(-dt / tau_divb);
  // clang-format on
}

template <typename Real, typename Source>
__global__ void
update_ei_kernel(FieldsView<Real> qq_orgn, FieldsView<Real> qq_argm,
                 FieldsView<Real> qq_rslt, Array3DView<Real> pr,
                 Array3DView<Real> bb, Array3DView<Real> ht, Array3DView<Real> vb,
                 Source source, GridView<Real> grid, Real dt) {
  int i, j, k;
  if (!compute_index_within_margin(i, j, k, grid))
    return;

  // Et: total energy per unit volume
  // ei: is the internal energy per unit mass
  // clang-format off
  const Real Et =
      +qq_orgn.ro(i, j, k) * qq_orgn.ei(i, j, k) +
      0.5 * qq_orgn.ro(i, j, k) *
          (+ qq_orgn.vx(i, j, k) * qq_orgn.vx(i, j, k)
           + qq_orgn.vy(i, j, k) * qq_orgn.vy(i, j, k)
           + qq_orgn.vz(i, j, k) * qq_orgn.vz(i, j, k)) +
      pii8<Real> *
          (+ qq_orgn.bx(i, j, k) * qq_orgn.bx(i, j, k)
           + qq_orgn.by(i, j, k) * qq_orgn.by(i, j, k)
           + qq_orgn.bz(i, j, k) * qq_orgn.bz(i, j, k));

  qq_rslt.ei(i, j, k) =
      (Et +
       dt * (- space_centered_4th(ht, qq_argm.vx, grid.dxi[i], i, j, k, grid.is, 0, 0, grid)
             - space_centered_4th(ht, qq_argm.vy, grid.dyi[j], i, j, k, 0, grid.js, 0, grid)
             - space_centered_4th(ht, qq_argm.vz, grid.dzi[k], i, j, k, 0, 0, grid.ks, grid)
             + pii4<Real> * (+ space_centered_4th(vb, qq_argm.bx, grid.dxi[i], i, j, k, grid.is, 0, 0, grid)
                             + space_centered_4th(vb, qq_argm.by, grid.dyi[j], i, j, k, 0, grid.js, 0, grid)
                             + space_centered_4th(vb, qq_argm.bz, grid.dzi[k], i, j, k, 0, 0, grid.ks, grid)))
       - 0.5 * qq_rslt.ro(i, j, k) *
           (+ qq_rslt.vx(i, j, k) * qq_rslt.vx(i, j, k)
            + qq_rslt.vy(i, j, k) * qq_rslt.vy(i, j, k)
            + qq_rslt.vz(i, j, k) * qq_rslt.vz(i, j, k)) -
       pii8<Real> *
           (+ qq_rslt.bx(i, j, k) * qq_rslt.bx(i, j, k)
            + qq_rslt.by(i, j, k) * qq_rslt.by(i, j, k)
            + qq_rslt.bz(i, j, k) * qq_rslt.bz(i, j, k))
       + source.ei(qq_argm, i, j, k)
      ) / qq_rslt.ro(i, j, k);
  // clang-format on
}

/// @brief Dummy source class (without source terms)
/// @details Volumetric heat / force terms are expected.
template <typename Real> struct NoSource {
  /// External force: x-direction
  __host__ __device__ inline Real vx(const FieldsView<Real> &, int, int,
                                     int) const noexcept {
    return 0.0;
  }

  /// External force: y-direction
  __host__ __device__ inline Real vy(const FieldsView<Real> &, int, int,
                                     int) const noexcept {
    return 0.0;
  }

  /// External force: z-direction
  __host__ __device__ inline Real vz(const FieldsView<Real> &, int, int,
                                     int) const noexcept {
    return 0.0;
  }

  /// External heating
  __host__ __device__ inline Real ei(const FieldsView<Real> &, int, int,
                                     int) const noexcept {
    return 0.0;
  }
};

template <typename Real, typename BoundaryCondition, typename EOS,
          typename Source = NoSource<Real>>
struct Integrator {
  /// @brief CUDA kernel shape
  cuda::KernelShape3D &cu_shape;

  /// @brief Spatial grid
  Grid<Real, backend::CUDA> &grid;
  /// @brief Equation of states
  EOS eos;
  /// @brief MHD state
  Fields<Real, backend::CUDA> &qq;
  /// @brief Workspace
  Fields<Real, backend::CUDA> qq_argm, qq_rslt;

  /// @brief Halo exchanger
  HaloExchanger<Real, backend::CUDA> halo_exchanger;
  /// @brief Boundary condition for MHD equations
  BoundaryCondition bc;
  /// @brief Body source for MHD equations
  Source source;
  /// @brief Artificial viscosity for MHD equations
  ArtificialViscosity<Real, EOS> artdiff;

  /// @brief Workspace for timestep calculation
  TimeStep<Real> time_step;

  /// @brief gas pressure
  Array3D<Real, backend::CUDA> pr;
  /// @brief magnetic field strength bx*bx + by*by + bz*bz
  Array3D<Real, backend::CUDA> bb;
  /// @brief enthalpy + 2*magnetic energy + kinetic energy
  Array3D<Real, backend::CUDA> ht;
  /// @brief inner product of velocity and magnetic field vx*bx + vy*by + vz*bz
  Array3D<Real, backend::CUDA> vb;

  /// @brief CFL number
  Real cfl_number;
  /// @brief propagation speed fo divergence B
  Real ch_divb;
  /// @brief square of ch_divb;
  Real ch_divb_square;
  /// @brief damping time scape for divergence B
  Real tau_divb;

  Integrator(Config &config, Fields<Real, backend::CUDA> &qq,
             Grid<Real, backend::CUDA> &grid, ExecContext<backend::CUDA> &exec_ctx)
      : cu_shape(exec_ctx.cu_shape), grid(grid), eos(config), qq(qq),
        qq_argm(grid), qq_rslt(grid), halo_exchanger(grid, exec_ctx), bc(config),
        artdiff(config, grid, eos, exec_ctx.cu_shape),
        time_step(exec_ctx.cu_shape),
        pr(grid.i_total, grid.j_total, grid.k_total),
        bb(grid.i_total, grid.j_total, grid.k_total),
        ht(grid.i_total, grid.j_total, grid.k_total),
        vb(grid.i_total, grid.j_total, grid.k_total) {
    cfl_number = config["mhd"]["cfl_number"].template as<Real>();
  }

  /// @brief Update MHD equations using 4th order space-centered scheme
  void update_sc4(Fields<Real, backend::CUDA> &qq_orgn,
                  Fields<Real, backend::CUDA> &qq_argm,
                  Fields<Real, backend::CUDA> &qq_rslt, const Real dt) {
    pr_bb_ht_vb_kernel<Real><<<cu_shape.grid_dim, cu_shape.block_dim>>>(
        qq_argm.view(), pr.view(), bb.view(), ht.view(), vb.view(), grid.view(),
        eos.gm);
    MISO_CUDA_CHECK(cudaGetLastError());

    update_ro_kernel<Real><<<cu_shape.grid_dim, cu_shape.block_dim>>>(
        qq_orgn.view(), qq_argm.view(), qq_rslt.view(), grid.view(), dt);
    MISO_CUDA_CHECK(cudaGetLastError());

    update_vx_kernel<Real><<<cu_shape.grid_dim, cu_shape.block_dim>>>(
        qq_orgn.view(), qq_argm.view(), qq_rslt.view(), pr.view(), bb.view(),
        source, grid.view(), dt);
    MISO_CUDA_CHECK(cudaGetLastError());

    update_vy_kernel<Real><<<cu_shape.grid_dim, cu_shape.block_dim>>>(
        qq_orgn.view(), qq_argm.view(), qq_rslt.view(), pr.view(), bb.view(),
        source, grid.view(), dt);
    MISO_CUDA_CHECK(cudaGetLastError());

    update_vz_kernel<Real><<<cu_shape.grid_dim, cu_shape.block_dim>>>(
        qq_orgn.view(), qq_argm.view(), qq_rslt.view(), pr.view(), bb.view(),
        source, grid.view(), dt);
    MISO_CUDA_CHECK(cudaGetLastError());

    update_bx_kernel<Real><<<cu_shape.grid_dim, cu_shape.block_dim>>>(
        qq_orgn.view(), qq_argm.view(), qq_rslt.view(), grid.view(), dt);
    MISO_CUDA_CHECK(cudaGetLastError());

    update_by_kernel<Real><<<cu_shape.grid_dim, cu_shape.block_dim>>>(
        qq_orgn.view(), qq_argm.view(), qq_rslt.view(), grid.view(), dt);
    MISO_CUDA_CHECK(cudaGetLastError());

    update_bz_kernel<Real><<<cu_shape.grid_dim, cu_shape.block_dim>>>(
        qq_orgn.view(), qq_argm.view(), qq_rslt.view(), grid.view(), dt);
    MISO_CUDA_CHECK(cudaGetLastError());

    update_ph_kernel<Real><<<cu_shape.grid_dim, cu_shape.block_dim>>>(
        qq_orgn.view(), qq_argm.view(), qq_rslt.view(), ch_divb_square, tau_divb,
        grid.view(), dt);
    MISO_CUDA_CHECK(cudaGetLastError());

    update_ei_kernel<Real><<<cu_shape.grid_dim, cu_shape.block_dim>>>(
        qq_orgn.view(), qq_argm.view(), qq_rslt.view(), pr.view(), bb.view(),
        ht.view(), vb.view(), source, grid.view(), dt);
    MISO_CUDA_CHECK(cudaGetLastError());
  }

  /// @brief Runge-Kutta 4th order time integration step
  void runge_kutta_4step(const Real dt) {
    update_sc4(qq, qq, qq_rslt, dt / 4.0);
    qq_argm.copy_from(qq_rslt);
    bc.apply(qq_argm.view());
    halo_exchanger.apply(qq_argm);

    update_sc4(qq, qq_argm, qq_rslt, dt / 3.0);
    qq_argm.copy_from(qq_rslt);
    bc.apply(qq_argm.view());
    halo_exchanger.apply(qq_argm);

    update_sc4(qq, qq_argm, qq_rslt, dt / 2.0);
    qq_argm.copy_from(qq_rslt);
    bc.apply(qq_argm.view());
    halo_exchanger.apply(qq_argm);

    update_sc4(qq, qq_argm, qq_rslt, dt);
    qq.copy_from(qq_rslt);
    bc.apply(qq.view());
    halo_exchanger.apply(qq);
  }

  /// @brief Apply artificial viscosity
  void apply_artificial_viscosity(const Real dt) {
    artdiff.characteristic_velocity_eval(qq);

    // x direction
    artdiff.update(qq, qq_rslt, Direction::X, dt);
    qq.copy_from(qq_rslt);
    bc.apply(qq.view());
    halo_exchanger.apply(qq);

    // y direction
    artdiff.update(qq, qq_rslt, Direction::Y, dt);
    qq.copy_from(qq_rslt);
    bc.apply(qq.view());
    halo_exchanger.apply(qq);

    // z direction
    artdiff.update(qq, qq_rslt, Direction::Z, dt);
    qq.copy_from(qq_rslt);
    bc.apply(qq.view());
    halo_exchanger.apply(qq);
  }

  /// @brief Calculate time spacing based on CFL condition
  Real cfl() const {
    cfl_kernel<Real>
        <<<cu_shape.grid_dim, cu_shape.block_dim, time_step.shared_mem_size>>>(
            qq.view(), grid.view(), time_step.min_values_device, cfl_number,
            eos.gm);
    MISO_CUDA_CHECK(cudaGetLastError());
    MISO_CUDA_CHECK(cudaDeviceSynchronize());

    time_step.copy_to_host();
    auto dt = *std::min_element(time_step.min_values_host,
                                time_step.min_values_host + time_step.n_blocks);
    auto dt_max =
        *std::max_element(time_step.min_values_host,
                          time_step.min_values_host + time_step.n_blocks);
    Real dt_global;
    MPI_Allreduce(&dt, &dt_global, 1, mpi::data_type<Real>(), MPI_MIN,
                  mpi::comm());
    return dt_global;
  }

  /// @brief Set parameters for divergence B cleaning
  void divb_parameters_set(const Real dt) {
    ch_divb = 0.8 * cfl_number * grid.min_dxyz / dt;
    ch_divb_square = ch_divb * ch_divb;
    tau_divb = 2.0 * dt;
  }

  /// @brief Update MHD equations by one time step
  void update(const Real dt) {
    divb_parameters_set(dt);
    runge_kutta_4step(dt);
    apply_artificial_viscosity(dt);
  }

  // Prohibit copy and move
  Integrator(const Integrator &) = delete;
  Integrator &operator=(const Integrator &) = delete;
  Integrator(Integrator &&) = delete;
  Integrator &operator=(Integrator &&) = delete;
};

}  // namespace impl_cuda
}  // namespace mhd
}  // namespace miso
